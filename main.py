import math
import imutils
import cv2

img = cv2.imread('photo1.jpg') #returns numpy array of bgr numbers.
if img.shape[1] > 800:
    img = imutils.resize(img, width=800)
grey_format_photo = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("Grey format photo", grey_format_photo)
grey_format_photo = cv2.adaptiveThreshold(grey_format_photo, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imshow("Adaptive Gaussian Thresholding", grey_format_photo)
grey_format_photo = cv2.GaussianBlur(grey_format_photo, (21, 21), cv2.BORDER_DEFAULT)
cv2.imshow("Gaussian Blur photo", grey_format_photo)
cv2.imwrite("grey_format_photo.jpg", grey_format_photo)
canny_edged_photo = cv2.Canny(grey_format_photo, 35, 35)
cv2.imshow("Canny edged photo", canny_edged_photo)
cv2.imwrite("canny_edged_photo.jpg", canny_edged_photo)
kernel_matrix = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
morphological_closing = cv2.morphologyEx(canny_edged_photo, cv2.MORPH_CLOSE, kernel_matrix)
cv2.imwrite("morphclosing.jpg", morphological_closing)
cv2.imshow("Closed contours image", morphological_closing)
finded_contours = cv2.findContours(morphological_closing.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[1]

rectangles = 0
circles = 0

for contour in finded_contours:
    square = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    approx_contour = cv2.approxPolyDP(contour, 0.04 * perim, True)
    if len(approx_contour) == 4:
        if square > 2000:
            rectangles += 1
            cv2.drawContours(img, [approx_contour], -1, (0, 0, 255), 5)
            continue
    if square > 500:
        if perim == 0:
            break
        circ = 4 * math.pi * (square / (perim * perim))
        if 0.8 < circ < 1.2:
            circles += 1;
            cv2.drawContours(img, [contour], -1, (255, 0, 0), 4)
circles = int(circles / 2)
rectangles = int(rectangles / 2)
print("Rectangles:",rectangles, "   Circles:",circles)
cv2.imshow("Image with drawed contours", img)
cv2.imwrite("lastimage.jpg", img)
cv2.waitKey(0)