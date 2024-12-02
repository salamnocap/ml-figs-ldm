import site
import os

path = site.getsitepackages()[0]
file_path = os.path.join(path, 'albumentations', 'core', 'bbox_utils.py')

if not os.path.exists(file_path):
    print(f"Error: {file_path} not found.")
else:
    with open(file_path, 'r') as file:
        code = file.read()

    old_code = """
    # Check if all values are in range [0, 1]
    in_range = (bboxes[:, :4] >= 0) & (bboxes[:, :4] <= 1)
    close_to_zero = np.isclose(bboxes[:, :4], 0)
    close_to_one = np.isclose(bboxes[:, :4], 1)
    valid_range = in_range | close_to_zero | close_to_one

    if not np.all(valid_range):
        invalid_idx = np.where(~np.all(valid_range, axis=1))[0][0]
        invalid_bbox = bboxes[invalid_idx]
        invalid_coord = ["x_min", "y_min", "x_max", "y_max"][np.where(~valid_range[invalid_idx])[0][0]]
        invalid_value = invalid_bbox[np.where(~valid_range[invalid_idx])[0][0]]
        raise ValueError(
            f"Expected {invalid_coord} for bbox {invalid_bbox} to be in the range [0.0, 1.0], got {invalid_value}.",
        )

    # Check if x_max > x_min and y_max > y_min
    valid_order = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])

    if not np.all(valid_order):
        invalid_idx = np.where(~valid_order)[0][0]
        invalid_bbox = bboxes[invalid_idx]
        if invalid_bbox[2] <= invalid_bbox[0]:
            raise ValueError(f"x_max is less than or equal to x_min for bbox {invalid_bbox}.")

        raise ValueError(f"y_max is less than or equal to y_min for bbox {invalid_bbox}.")
    """
    
    new_code = """
    # Check if all values are in range [0, 1]
    for bbox in bboxes:
        if bbox[0] == bbox[2]:
            bbox[0] -= 0.01
            bbox[2] += 0.01
        if bbox[1] == bbox[3]:
            bbox[1] -= 0.01
            bbox[3] += 0.01
        
        bbox[0] = max(0.0, min(bbox[0], 1.0)) 
        bbox[1] = max(0.0, min(bbox[1], 1.0))
        bbox[2] = max(0.0, min(bbox[2], 1.0))
        bbox[3] = max(0.0, min(bbox[3], 1.0))

    in_range = (bboxes[:, :4] >= 0) & (bboxes[:, :4] <= 1)
    close_to_zero = np.isclose(bboxes[:, :4], 0)
    close_to_one = np.isclose(bboxes[:, :4], 1)
    valid_range = in_range | close_to_zero | close_to_one

    if not np.all(valid_range):
        invalid_idx = np.where(~np.all(valid_range, axis=1))[0][0]
        invalid_bbox = bboxes[invalid_idx]
        invalid_coord = ["x_min", "y_min", "x_max", "y_max"][np.where(~valid_range[invalid_idx])[0][0]]
        invalid_value = invalid_bbox[np.where(~valid_range[invalid_idx])[0][0]]
        raise ValueError(
            f"Expected {invalid_coord} for bbox {invalid_bbox} to be in the range [0.0, 1.0], got {invalid_value}.",
        )

    # Check if x_max > x_min and y_max > y_min
    valid_order = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])

    if not np.all(valid_order):
        invalid_idx = np.where(~valid_order)[0][0]
        invalid_bbox = bboxes[invalid_idx]
        if invalid_bbox[2] <= invalid_bbox[0]:
            raise ValueError(f"x_max is less than or equal to x_min for bbox {invalid_bbox}.")

        raise ValueError(f"y_max is less than or equal to y_min for bbox {invalid_bbox}.")
    """

    modified_code = code.replace(old_code, new_code)

    with open(file_path, 'w') as file:
        file.write(modified_code)

    print(f"File {file_path} has been successfully updated.")