import pytest
from astropy.table import Table

from exosignal.errors import InvalidTicError, NoTessDataError
from exosignal.tess import discover_spoc_products, parse_tic_id, selected_product_indices


@pytest.mark.parametrize("raw, expected", [("TIC 307210830", 307210830), ("307210830", 307210830)])
def test_parse_tic_id(raw, expected):
    assert parse_tic_id(raw) == expected


def test_parse_tic_id_rejects_other_identifiers():
    with pytest.raises(InvalidTicError):
        parse_tic_id("TOI 123")


def test_standard_cadence_is_selected_once_per_sector():
    class SearchResult:
        table = Table(
            {"sequence_number": [4, 31, 31], "exptime": [120.0, 20.0, 120.0]}
        )

    assert selected_product_indices(SearchResult()) == [0, 2]


def test_no_mast_products_is_a_clear_error(monkeypatch):
    monkeypatch.setattr("exosignal.tess.lk.search_lightcurve", lambda *args, **kwargs: [])
    with pytest.raises(NoTessDataError, match="No SPOC TESS light curves"):
        discover_spoc_products(123)
