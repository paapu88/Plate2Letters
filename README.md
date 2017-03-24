# Plate2Letters
Given a licence plate, produces the alphabets and the numbers on it to output

At the moment the licence plate must be BIG in the video to become recognized
( i did not realize that resolution in a camera and in a video recorder differ so much)...

Usage
python3 ./showVideo.py ./veetiauto1.mp4 False 5

1st argument: the filename of the video
2nd argument: set True if you need to rotate the image
3rd argument: how many frames are skipped when reading

Push start at QT window

By ffmpeg one gets frame by frame:
ffmpeg -i Rekisteri_4_short.mp4 -r 1/1 %03d.jpg
and to check resolution:
python3 ~/PycharmProjects/Plate2Letters/showPictureMatplotlib.py 001.jpg
by zooming it seem the minimum resolution is 8x12, try also 10x15


