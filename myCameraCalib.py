import numpy as np
import cv2
import glob

#dimenzije ploče
xOs = 8
yOs = 6

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((xOs * yOs, 3), np.float32)
objp[:, :2] = np.mgrid[0:xOs, 0:yOs].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

#učitavanje slika
images = glob.glob('calibration_wide/GO*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (xOs, yOs), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (xOs, yOs), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)

#ostatak koda
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

#pohrana parametara unutar file-a
np.savez('calibParam.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

#test
img = cv2.imread('calibration_wide/GOPR0041.jpg')
h, w = img.shape[:2]

# alpha = 0 min neželjenih pixela alpha = 1 max
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))

#prva vrsta ispravljanja slike
# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
#pohrana slike
cv2.imwrite('calibresult.png', dst)

#prikaz slike na ekranu
cv2.imshow('chess board', dst)
cv2.waitKey(0)

# #druga vrsta ispravljanja slike
# # undistort
# mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
# dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

# # crop the image
# x, y, w, h = roi
# dst = dst[y:y+h, x:x+w]
##pohrana slike
# cv2.imwrite('calibresult.png', dst)

# #prikaz slike na ekranu
# cv2.imshow('chess board', dst)
# cv2.waitKey(0)

totalError = 0
for i in range(len(objpoints)):
    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
    error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2) / len(imgpoints2)
    totalError += error

print ("total error: ", totalError / len(objpoints))

cv2.destroyAllWindows()