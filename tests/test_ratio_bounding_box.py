from mouse_pointer_controller.utils import RatioPoint, RatioBoundingBox

def test_ratio_bounding_box_size():
    assert RatioBoundingBox(top_left=RatioPoint(.25, .25), bottom_right=RatioPoint(.75, .75)).size == .25