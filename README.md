

# atrous-cGAN-for-SAR-optical-synthesis

Implementation of the atrous cGAN and pix2pix for the SAR-optical translation oriented to tow applications: crop recognition and deforestation detection.

## Dependencies

- Python 3.7
- Numpy 1.16.1
- Tensorflow 1.14.1

## Datasets
Land covers in Brazil.
- Crop recognition:
                  Campo Verde (CV), Mato Grosso state. 
                  Luis Eduardo Magalhães (LEM), Bahia state.
- Deforestation detection:
                  Amazon Rainforest (AR), Rondônia state

## Architecture
The size of the optical image patches was set to 256 x 256 pixels for CV and LEM datasets, and 128 x 128 pixels for the AR. 
The SAR image patches were 3 times bigger than their optical counterpart because of the difference of resolution between the two sensors. Accordingly, to use the SAR patches with the same size of optical, they were downsampled using an aditional convolutional layer with stride=3. The following table shows a detailed configuration of the parameters.



![Architecture](https://github.com/jnoat92/atrous-cGAN-for-SAR-optical-synthesis/blob/master/Architecture.png)



