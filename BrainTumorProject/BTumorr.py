# import numpy as np
# # -------------------------------
# # PREDICTION (CORRECT WAY)
# # -------------------------------
# img_path = '/home/shreevani_ms7/MainProject/dataset/Testing/pituitary/Te-pi_7.jpg'

# img = cv2.imread(img_path)

# # show original image
# plt.imshow(img)
# plt.title("Test Image")
# plt.axis("off")
# plt.show()

# # preprocess before prediction
# img = preprocess(img)
# img = img.reshape(1,128,128,3)

# prediction = model.predict(img)

# print("\nPrediction probabilities:", prediction)
# print("Predicted class:", labels[np.argmax(prediction)])