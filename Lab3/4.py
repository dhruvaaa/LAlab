import numpy as np
import matplotlib.pyplot as plt


"""
Classification is fundamentally difficult because there can be
variations within the same class and similarities between different classes.

Intra-class variation means that samples belonging to the same class
can have different feature values.

Inter-class variation means that samples belonging to different classes
can have similar or overlapping feature values.

Because of this feature overlap, it can be difficult to separate
the two classes perfectly.
"""


# Function to extract the mean pixel value
def feature_extractor(image):
    feature = np.mean(image)
    return feature


# Create some simulated 2D images
image1 = np.random.normal(100, 10, (10, 10))
image2 = np.random.normal(150, 10, (10, 10))

# Extract one feature from each image
feature1 = feature_extractor(image1)
feature2 = feature_extractor(image2)

print("Feature from class 1:", feature1)
print("Feature from class 2:", feature2)


# Generate features for two classes
class1 = np.random.normal(100, 10, 50)
class2 = np.random.normal(130, 10, 50)


# Plot histogram
plt.hist(class1, bins=10, alpha=0.5, label="Class 1")
plt.hist(class2, bins=10, alpha=0.5, label="Class 2")

plt.xlabel("Extracted Feature")
plt.ylabel("Number of Samples")
plt.legend()
plt.show()
