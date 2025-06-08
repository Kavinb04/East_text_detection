import pytesseract
import cv2

img = cv2.imread(r'test.jpg')

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
text = pytesseract.image_to_string(img)
print(text)






