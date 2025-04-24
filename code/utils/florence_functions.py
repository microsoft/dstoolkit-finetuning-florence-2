from PIL import ImageDraw, ImageFont
import numpy as np
import matplotlib.pyplot as plt

def draw_bounding_boxes(image, combined_results, size_bb = 20, size_text = 20, label_bb = True, color_bb = True):
    """
    Draw bounding box for dictionary of bounding boxes and labels.
    {"bboxes": all_boxes, "labels": all_labels} OR {"polygons": all_boxes, "labels": all_labels}
    size_bb - width of bounding box
    label_bb - add label to image bb
    color_bb - color the bounding box with random different color per class, else red
    """
    # Draw bounding boxes on the image
    modified_image = image.copy()
    draw = ImageDraw.Draw(modified_image)
    font = ImageFont.truetype("arial.ttf", size_text)
    classes = list(np.unique(combined_results['labels']))
    
    bbox_type = "bboxes" if "bboxes" in combined_results else "polygons" # get bbox type from combined_results
    
    cmap_type = 'RdYlBu'
    if color_bb:
        colors = plt.get_cmap(cmap_type, len(classes))
    for box, label in zip(combined_results[bbox_type], combined_results["labels"]):
        if color_bb:
            class_index = classes.index(label)
            color = tuple(int(c * 255) for c in colors(class_index)[:3])
        else:
            color = "red"
        
        x1, y1 = box[0], box[1]
        if bbox_type == "bboxes":
            draw.rectangle(box, outline=color, width=size_bb)
        elif bbox_type == "polygons": # quad_boxes - number of points: x1, y1, x2, y2, ...., xi, yi
            draw.polygon(box, outline=color, fill=color, width=size_bb)
        if label_bb:
            draw.text((x1+5, y1+5), label, fill="black", font=font)
            
    fig, ax = plt.subplots(1)
    ax.imshow(modified_image)
    plt.axis('off')
    plt.show()
    
def inference_florance(image, task_input, text_inputs:list=[], florance2_setup:list=None):
    """
    Perform a variety of tasks for Florance-2 model. 
    Prompts take priority over task_input and text_inputs.
    If no prompts are given, we can use the default task_input with suplimentatry text inputs.
    Text inputs are used to add more context to the task_input, such as bounding boxes.
    """
    # we can combine as so: 
    # prompts = "detect cars and motobikes in the image"
    # but if it cannot detect the object, it may be because of model's limitations in handling complex scenes with multiple objects. 
    # Therefore, its better to call one at a time to get the bounding boxes for each class
    model, processor, device, torch_dtype = florance2_setup
    
    if len(text_inputs)>0: 
        prompts = [task_input + text_input for text_input in text_inputs]
    else:
        prompts = [task_input]
    print(f"Prompts: {prompts}")
    
    model_response = []
    for prompt in prompts:
        # Process the input
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
        # print(generated_text)

        # Post-process the output
        parsed_answer = processor.post_process_generation(generated_text, task=task_input, image_size=(image.width, image.height))
        
        # Collect results
        model_response.append(parsed_answer)
    
    return model_response