import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./vision.json"

import base64

# Imports the Google Cloud client library
from google.cloud import vision


def encode_image(image):
    with open(image, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    return encoded_string


def run_quickstart() -> vision.EntityAnnotation:
    client = vision.ImageAnnotatorClient()
    file_uri = "./images/th_th_pixta_77370244_M-2000x1333（中）.jpeg"

    with open(file_uri, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    # image.source.image_uri = file_uri

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations

    print("Labels:")
    for label in labels:
        print(label)
        print(label.description)

    return labels


def localize_objects(path):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    from google.cloud import vision

    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()
    image = vision.Image(content=content)

    objects = client.object_localization(image=image).localized_object_annotations

    print(f"Number of objects found: {len(objects)}")
    for object_ in objects:
        print(f"\n{object_.name} (confidence: {object_.score})")
        print("Normalized bounding polygon vertices: ")
        for vertex in object_.bounding_poly.normalized_vertices:
            print(f" - ({vertex.x}, {vertex.y})")


def detect_web(path):
    client = vision.ImageAnnotatorClient()

    with open(path, "rb") as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    response = client.web_detection(image=image)
    annotations = response.web_detection

    if annotations.web_entities:
        print("\n{} Web entities found: ".format(len(annotations.web_entities)))

        for entity in annotations.web_entities:
            print(f"\n\tScore      : {entity.score}")
            print(f"\tDescription: {entity.description}")

    if response.error.message:
        raise Exception(
            "{}\nFor more info on error messages, check: "
            "https://cloud.google.com/apis/design/errors".format(response.error.message)
        )


if __name__ == "__main__":
    # run_quickstart()
    # localize_objects("images/IMG_1620（中）.jpeg")
    detect_web("images/IMG_1620（中）.jpeg")
