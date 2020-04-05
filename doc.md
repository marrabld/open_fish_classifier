## Template matching
**commit: [f5ec03be9a1e442b8542e434d0388582682404e6](https://github.com/marrabld/open_fish_classifier/tree/f5ec03be9a1e442b8542e434d0388582682404e6)**

The first attempt to track fish through multiple frames was done using naive template matching.
Template matching essentially tries to find a reference image inside a larger image.
Using the existing bounding boxes provided by ecologists, we attempted to track these fish through subsequent frames using [OpenCV's `matchTemplate()`](https://docs.opencv.org/2.4/modules/imgproc/doc/object_detection.html?highlight=matchtemplate).

Results (0.25x speed):

* https://gfycat.com/obedientnegativebackswimmer
* https://gfycat.com/uniqueuntidyfrigatebird
* https://gfycat.com/satisfiedadeptaxisdeer


On its own, this approach did not work particularly well. `matchTemplate()` considers every pixel in the reference image as important as any others. 
We had hoped to mitigate this issue by updating the reference image each frame to hopefully keep the fish as the main content of the reference image.
While this worked in some cases, it also had the opposite effect in others; if the initial match identified a section of coral as the best match for the reference image, the reference image would be updated with more coral and the cycle would continue.

In addition the template matching did not cope well with fish changing direction, and was very slow to process each frame.
Overall this approach seemed to lose / mistake the original template within a very short number of frames and did not seem like a great option.

## Optical Flow Tracking
**commit: [da4025d5e5b0bd9ddac90980ecc1803709d5fe06](https://github.com/marrabld/open_fish_classifier/tree/da4025d5e5b0bd9ddac90980ecc1803709d5fe06)**

The next approach was to use the [Lucas-Kanade Optical Flow Tracking method](https://en.wikipedia.org/wiki/Lucas%E2%80%93Kanade_method). Again, this was an attempt to track fish in predefined bounding boxes through subsequent frames. 

Optical Flow Tracking is achieved by identifying a few key, interesting points within a frame and attempting to determine their location in the next frame, allowing you to determine a motion vector between frames for each point.

In this case we used the bounding boxes to identify the initial position of fish we were interested in and  within each of these boxes we were able to determine some useful points to track. Determining a 'useful' point can be done in a variety of way such as sampling edges found with Canny edge detection, or simply using OpenCV's `goodFeaturesToTrack()`.

As the points are tracked across each frame, the bounding box's position is updated according to the mean motion vector of all points. Points are re-sampled when they move outside of the current bounding box.

Results (0.25x speed)
* https://gfycat.com/unitedfantastickoodoo
* https://gfycat.com/fatalwildacaciarat
* https://gfycat.com/infatuatedesteemedibisbill
* https://gfycat.com/agitatedgrotesquedoctorfish

There are a large number of input parameters that can be tweaked using this method so additional work is needed to determine the best combinations for each input video.

There is also currently no real outlier detection to handle points that were not tracked correctly, these cause the mean motion vector to be inaccurate. In addition, the original bounding box is simply translated by the motion vector and never scaled or resized, this causes issues when resampling as the sample area may no longer be representative of the fish's relative size / orientation.

Overall this solution seems promising, and I believe if we can combine it with some better bounding box detection to allow the initial bounding box to be dynamically resized the accuracy would improve immensely.

## Background subtraction
**commit: [692c596c60800b3e14fde70c2f340dd8d46ebdc8](https://github.com/marrabld/open_fish_classifier/tree/692c596c60800b3e14fde70c2f340dd8d46ebdc8)**

In an attempt to dynamically locate bounding boxes, background subtraction was used.
Background subtraction is the process by which the background is identified and removed from the image, leaving only the objects in the foreground for processing.

There a number of different ways to perform background subtraction, some as simple as computing the difference between a reference 'background only' frame and the current frame, all the way up to more complex algorithms like KNN and MOG2.

The hopes with background subtraction is that it removes enough of the static background to allow for reasonably accurate edge detection of the foreground objects; allowing us to draw accurate bounding boxes around each object (fish) in the foreground.

As with Optical Flow, there is a large number of pre- and post-processing steps that can be performed, as well as a large combination of input parameters that can be tweaked to alter the results. 

Results (0.25x speed)
* https://gfycat.com/adorablesecondhandcorydorascatfish
* https://gfycat.com/occasionalinfamousbat
* https://gfycat.com/glisteningmeaslyauk
* https://gfycat.com/clearcutshamefulgermanshepherd
* https://gfycat.com/infamousjointhind

These results were taken using the KNN background subtraction method, applying an erosion + dilation morphology (kernel=10x10) to remove background noise.

While the results are not quite as accurate as we would like just yet (many fish are not detected, or detected in multiple sections), I believe there is still hope that it can be tweaked to improve the accuracy.

## SiamMask
**commit:** not yet featured

[SiamMask](https://arxiv.org/abs/1812.05050) is a visual object tracker that utilises the combination of Siamese Networks and Binary Segmentation (Masking).

Although SiamMask's default models and configurations were not trained on fish data, the initial attempts at tracking fish in out training videos were promising.

Results (0.25x speed)
* https://gfycat.com/majesticunripefeline

Out of the box, the [provided demo code](https://github.com/foolwood/SiamMask) for SiamMask only supports tracking of a single region of interest (ROI), whereas we need to be able to track multiple fish across each frame.

We made a naive attempt to modify the demo code to support tracking multiple objects. While it seemed to be working well initially, once we had added tracking to all the available bounding boxes there was a very noticable drop off in accuracy for all the objects being tracked.

Results (0.25x speed)
* https://gfycat.com/potableunnaturalislandcanary

Notice that the quality of the tracking of the fish from the first example has deteriorated significantly when tracking the rest of the bounding boxes simultaneously. It appears that more time needs to be spent familiarising with the SiamMask internals to correctly update it to support the tracking of multiple objects.