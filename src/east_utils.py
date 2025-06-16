# east_utils.py

import numpy as np
import cv2

def decode_east_predictions(scores, geometry, score_thresh=0.5):
    """
    Decodes the scores and geometry outputs from the EAST text detector
    into bounding box locations on image and confidence scores for each pixel

    Args:
        scores (np.array): output score map from the EAST model. 
            Shape: (1, 1, H, W)

        geometry (np.array): output geometry map from the EAST model. Stores every pixel's distance to each boundary
            Shape: (1, 5, H, W) 

        score_thresh (float): Confidence threshold for filtering weak text detections.

    Returns:
        boxes (list): List of rotated bounding boxes (x, y, w, h, angle)
        confidences (list): List of confidence scores corresponding to each box.
    """
    # Grab the dimensions of the scores volume (height and width)
    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # Loop over the number of rows
    for y in range(0, numRows):
        # Extract the scores and geometrical data for current row
        scoresData = scores[0, 0, y]
        dist_to_top = geometry[0, 0, y]
        dist_to_right = geometry[0, 1, y]
        dist_to_bottom = geometry[0, 2, y]
        dist_to_left = geometry[0, 3, y]
        anglesData = geometry[0, 4, y]

        # Loop over the cols
        for x in range(0, numCols):
            # Ignore if score less than threshold
            if scoresData[x] < score_thresh:
                continue

            # Compute the offset factor 
            # our resulting feature maps will be 4x smaller than the input image
            (offsetX, offsetY) = (x * 4.0, y * 4.0)

            # compute sin and cosine for the rotation angle
            angle = anglesData[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Calculate height and width of box
            h = dist_to_top[x] + dist_to_bottom[x]
            w = dist_to_right[x] + dist_to_left[x]

            # Compute both the starting and ending (x, y)-coordinates for
            # the text prediction bounding box
            endX = int(offsetX + (cos * dist_to_right[x]) + (sin * dist_to_bottom[x]))
            endY = int(offsetY - (sin * dist_to_right[x]) + (cos * dist_to_bottom[x]))
            startX = int(endX - w)
            startY = int(endY - h)

            # Add the bounding box coordinates and probability score to output lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[x])

    return (rects, confidences)

def east_preprocessing(image, input_size):
    """
    Prepares image to be fed into EAST model. Resizes image dimensions to a multiple of 32 and returns image in
    blob format. 

    Args:
        image (np.array): The input image (BGR)
        input_size (tuple): The (width, height) tuple for the input to the EAST model

    Returns:
        np.array: The preprocessed image blob suitable for the EAST model.
    """

    # Resize the image to the input_size expected by the EAST model
    (h, w) = image.shape[:2]
    # Calculate the ratio of the original image to the new input size
    rW = w / float(input_size[0])
    rH = h / float(input_size[1])

    # Resize the image and create a blob
    resized_image = cv2.resize(image, input_size)
    
    # Extract blob from resized image
    blob = cv2.dnn.blobFromImage(resized_image, 1.0, input_size,
                                 (123.68, 116.78, 103.94), swapRB=True, crop=False)
    return blob, rW, rH

def adjust_boxes(boxes, rW, rH):
    """
    Adjusts bounding box coordinates back to the original image scale.

    Args:
        boxes (list): List of bounding boxes (startX, startY, endX, endY)
                      from the decoded EAST output.
        rW (float): Width ratio of original image to input_size.
        rH (float): Height ratio of original image to input_size.

    Returns:
        list: Adjusted bounding boxes (x, y, w, h) for the original image.
    """
    adjusted_boxes = []
    for (startX, startY, endX, endY) in boxes:
        # Scale the bounding box coordinates based on given ratios
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # Convert (startX, startY, endX, endY) to (x, y, w, h)
        x = startX
        y = startY
        w = endX - startX
        h = endY - startY
        adjusted_boxes.append((x, y, w, h))
    return adjusted_boxes

def apply_nms(boxes, confidences, nms_thresh):
    """
    Applies Non-Maximum Suppression to filter overlapping bounding boxes

    Args:
        boxes (list): List of bounding boxes (x, y, w, h)
        confidences (list): List of confidence scores
        nms_thresh (float): Non-Maximum Suppression threshold

    Returns:
        list: Indices of the boxes to keep after NMS
    """
    # Ensure boxes are in (x, y, x+w, y+h) format for NMSBoxes
    boxes_for_nms = [[x, y, x + w, y + h] for (x, y, w, h) in boxes]
    
    # Perform Non-Maximum Suppression
    indices = cv2.dnn.NMSBoxes(boxes_for_nms, confidences,
                               score_threshold=0.0, 
                               nms_threshold=nms_thresh)

    if len(indices) > 0:
        indices = indices.flatten()
    else:
        indices = []
    return indices