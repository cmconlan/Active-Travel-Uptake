"""Unit tests for the LTS rules. Run with: python -m pytest tests/"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))

from lts import as_list, classify, parse_speed_mph, provision_level  # noqa: E402
from config import LOCAL_20MPH_LTS  # noqa: E402


@pytest.mark.parametrize("tags, speed, expected", [
    # Traffic-free
    ({"highway": "cycleway"}, 20, 1),
    ({"highway": "path", "bicycle": "designated"}, 20, 1),
    ({"highway": "footway", "bicycle": "yes"}, 20, 1),
    ({"highway": "path"}, 20, 2),
    ({"highway": "pedestrian", "bicycle": "yes"}, 10, 1),
    ({"highway": "living_street"}, 10, 1),
    # Separated track on a main road
    ({"highway": "primary", "cycleway:both": "track"}, 30, 1),
    # Painted lanes
    ({"highway": "residential", "cycleway": "lane"}, 30, 2),
    ({"highway": "primary", "cycleway:both": "lane"}, 30, 3),
    ({"highway": "tertiary", "cycleway": "lane"}, 40, 3),
    ({"highway": "trunk", "cycleway": "lane"}, 50, 4),
    # Lane on one side only of a two-way road: the worse side wins -> mixed traffic
    ({"highway": "residential", "cycleway:left": "lane"}, 30, 3),
    # ...but on a one-way road the lane side is the one that matters
    ({"highway": "residential", "cycleway:left": "lane", "oneway": "True"}, 30, 2),
    # Shared bus lane
    ({"highway": "primary", "cycleway": "share_busway"}, 30, 3),
    ({"highway": "primary", "cycleway": "share_busway"}, 40, 4),
    # Mixed traffic
    ({"highway": "primary"}, 30, 4),
    ({"highway": "secondary_link"}, 30, 4),
    ({"highway": "tertiary"}, 30, 3),
    ({"highway": "residential"}, 30, 3),
    ({"highway": "residential"}, 20, LOCAL_20MPH_LTS),
    ({"highway": "residential", "lanes": "4"}, 30, 4),
    ({"highway": "unclassified"}, 40, 4),
    ({"highway": "service"}, 20, 2),
    # Sharrows and "separate" are not provision on this edge
    ({"highway": "residential", "cycleway": "shared_lane"}, 30, 3),
    ({"highway": "primary", "cycleway": "separate"}, 30, 4),
    # osmnx-simplified edges with list values: worst case
    ({"highway": "['residential', 'primary']"}, 30, 4),
    ({"highway": "residential", "cycleway": "['lane', 'no']"}, 30, 3),
])
def test_classify(tags, speed, expected):
    assert classify(tags, speed)[0] == expected


@pytest.mark.parametrize("raw, mph", [
    ("20 mph", 20), ("30", 30), ("GB:nsl_single", 60), ("50 km/h", pytest.approx(31.05)),
    ("['20 mph', '30 mph']", 30), ("signals", None), (None, None), (float("nan"), None),
])
def test_parse_speed(raw, mph):
    assert parse_speed_mph(raw) == mph


def test_as_list_handles_strings_lists_and_semicolons():
    assert as_list("lane") == ["lane"]
    assert as_list("['lane', 'track']") == ["lane", "track"]
    assert as_list("lane;track") == ["lane", "track"]
    assert as_list(None) == []


def test_two_way_track_on_both_sides():
    assert provision_level({"cycleway:left": "track", "cycleway:right": "track"}, oneway=False) == 3
