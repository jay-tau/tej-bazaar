from datetime import date
from pathlib import Path

import polars as pl
import pytest
import responses

from pipeline import delivery
from pipeline.delivery import DeliveryParseError, fetch_nse_delivery, parse_mto
from pipeline.fetch import BhavcopyFetchError, BhavcopyNotFoundError

# Trimmed from the real MTO_28102026.DAT (2019+ layout) and MTO_13062012.DAT
# (2010-2019 layout, CRLF); the 10,MTO line is rewritten to the kept records.
FIXTURE = Path(__file__).parent / "fixtures" / "nse_mto_20261028_sample.DAT"
FIXTURE_2012 = Path(__file__).parent / "fixtures" / "nse_mto_20120613_sample.DAT"


def test_parse_mto_schema():
    df = parse_mto(FIXTURE, date(2026, 10, 28))
    assert df.columns == ["date", "symbol", "series", "volume", "delivery_qty", "delivery_pct"]
    assert df.schema["date"] == pl.Date
    assert df.schema["volume"] == pl.Int64
    assert df.schema["delivery_qty"] == pl.Int64
    assert df.schema["delivery_pct"] == pl.Float64


def test_parse_mto_keeps_only_the_series_nse_carries():
    # The fixture also holds a YW debt row and an SM (SME) row; nse/ has neither.
    df = parse_mto(FIXTURE, date(2026, 10, 28))
    assert df["symbol"].to_list() == ["20MICRONS", "INFY", "RELIANCE", "SBIN", "TCS"]
    assert set(df["series"]) == {"EQ"}
    assert set(df["date"]) == {date(2026, 10, 28)}


def test_parse_mto_known_value_reliance():
    df = parse_mto(FIXTURE, date(2026, 10, 28))
    row = df.filter(pl.col("symbol") == "RELIANCE").row(0, named=True)
    assert row["volume"] == 21866598
    assert row["delivery_qty"] == 9900037
    assert row["delivery_pct"] == pytest.approx(45.27)


def test_parse_mto_older_layout_with_crlf():
    df = parse_mto(FIXTURE_2012, date(2012, 6, 13))
    assert df.height == 8
    row = df.filter(pl.col("symbol") == "3IINFOTECH").row(0, named=True)
    assert (row["volume"], row["delivery_qty"], row["delivery_pct"]) == (5514499, 3788329, 68.70)


def test_parse_mto_rejects_a_file_for_another_day():
    with pytest.raises(DeliveryParseError, match="header date 28102026"):
        parse_mto(FIXTURE, date(2026, 10, 29))


def test_parse_mto_rejects_a_short_file(tmp_path: Path):
    # A file cut off mid-download has fewer records than its header counts.
    short = tmp_path / "MTO_28102026.DAT"
    short.write_text("".join(FIXTURE.read_text().splitlines(keepends=True)[:-1]))
    with pytest.raises(DeliveryParseError, match="header says 7 records"):
        parse_mto(short, date(2026, 10, 28))


def test_parse_mto_rejects_a_truncated_final_percentage(tmp_path: Path):
    short = tmp_path / "MTO_28102026.DAT"
    short.write_text(FIXTURE.read_text()[:-2])  # 38.06 becomes 38.0, without a newline.
    with pytest.raises(DeliveryParseError, match="incomplete final line"):
        parse_mto(short, date(2026, 10, 28))


def test_parse_mto_rejects_a_wrong_total_with_the_right_count(tmp_path: Path):
    wrong = tmp_path / "MTO_28102026.DAT"
    wrong.write_text(FIXTURE.read_text().replace("024680455", "024680456"))
    with pytest.raises(DeliveryParseError, match="header says 7 records and 24680456 delivered"):
        parse_mto(wrong, date(2026, 10, 28))


def test_parse_mto_rejects_a_file_without_the_header_line(tmp_path: Path):
    html = tmp_path / "MTO_28102026.DAT"
    html.write_text("<!DOCTYPE html>\n<html></html>\n")
    with pytest.raises(DeliveryParseError, match="no 10,MTO header"):
        parse_mto(html, date(2026, 10, 28))


# --- fetch -------------------------------------------------------------


@responses.activate
def test_fetch_nse_delivery_happy_path(tmp_path: Path):
    body = FIXTURE.read_bytes()
    responses.add(
        responses.GET,
        delivery.NSE_MTO_URL.format(ddmmyyyy="28102026"),
        body=body,
        status=200,
        content_type="application/octet-stream",
    )
    out = fetch_nse_delivery(date(2026, 10, 28), tmp_path)
    assert out == tmp_path / "MTO_28102026.DAT"
    assert out.read_bytes() == body


@responses.activate
def test_fetch_nse_delivery_idempotent(tmp_path: Path):
    existing = tmp_path / "MTO_28102026.DAT"
    existing.write_bytes(b"cached")
    assert fetch_nse_delivery(date(2026, 10, 28), tmp_path) == existing
    assert len(responses.calls) == 0


@responses.activate
def test_fetch_nse_delivery_404_raises_notfound(tmp_path: Path):
    # What NSE answers before about 16:30 IST on the trade date, and on holidays.
    responses.add(
        responses.GET, delivery.NSE_MTO_URL.format(ddmmyyyy="28102026"), status=404
    )
    with pytest.raises(BhavcopyNotFoundError):
        fetch_nse_delivery(date(2026, 10, 28), tmp_path, retries=1)
    assert not list(tmp_path.glob("*.DAT"))


@responses.activate
def test_fetch_nse_delivery_asks_nse_about_a_special_weekend_session(tmp_path: Path):
    # Saturday 2024-01-20 traded and is in nse/, but the calendar misses it, so
    # the fetch leaves the choice of days to its caller.
    responses.add(
        responses.GET, delivery.NSE_MTO_URL.format(ddmmyyyy="20012024"), body=b"x", status=200
    )
    assert fetch_nse_delivery(date(2024, 1, 20), tmp_path).read_bytes() == b"x"


@responses.activate
def test_fetch_nse_delivery_gives_up_after_retries(tmp_path: Path):
    url = delivery.NSE_MTO_URL.format(ddmmyyyy="28102026")
    responses.add(responses.GET, url, status=503)
    responses.add(responses.GET, url, status=503)
    with pytest.raises(BhavcopyFetchError):
        fetch_nse_delivery(date(2026, 10, 28), tmp_path, retries=2, backoff_seconds=0.0)
    assert not list(tmp_path.glob("*.DAT"))
