import cv2
from motion import flow_to_magnitude, motion_mask_from_magnitude, extract_bounding_boxes


def process_farneback_frame(prev_gray, gray, frame, config):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        gray,
        None,
        config.pyr_scale,
        config.levels,
        config.winsize,
        config.iterations,
        config.poly_n,
        config.poly_sigma,
        config.flags,
    )

    magnitude = flow_to_magnitude(flow)
    mask = motion_mask_from_magnitude(magnitude, config.motion_threshold)
    boxes = extract_bounding_boxes(mask, config.min_contour_area)

    annotated = frame.copy()

    for (x, y, w, h) in boxes:
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.putText(
        annotated,
        "Method: Farneback",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    return annotated, boxes, mask, magnitude