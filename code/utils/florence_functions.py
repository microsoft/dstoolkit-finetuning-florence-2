import torch
from PIL import ImageDraw
import numpy as np
import matplotlib.pyplot as plt

    
def obj_det_florence(image, model, processor, classes=None, device="cpu", torch_dtype=torch.float32):
    """
    Return all the bounding boxes and labels for the classes in the image.
    """
    # we can combine as so: 
    # prompts = "detect cars and motobikes in the image"
    # but if it cannot detect the object, it may be because of model's limitations in handling complex scenes with multiple objects. 
    # Therefore, its better to call one at a time to get the bounding boxes for each class
    if classes is None:
        prompts = [None]
    else:
        prompts = [f"Locate the {i} in the image." for i in classes]

    all_boxes = []
    all_labels = []
    for prompt in prompts:
        # Process the input
        prompt = prompt.lower()
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, dtype=torch_dtype)

        # Generate predictions
        generated_ids = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024, # limit
            early_stopping=False,
            do_sample=False,
            num_beams=3,
        )

        # Decode the predictions
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

        # Post-process the output
        parsed_answer = processor.post_process_generation(generated_text, task="<OD>", image_size=(image.width, image.height))
        
        # Collect results
        all_boxes.extend(parsed_answer['<OD>']["bboxes"])
        all_labels.extend(parsed_answer['<OD>']["labels"])

    combined_results = {"bboxes": all_boxes, "labels": all_labels}
    
    return combined_results