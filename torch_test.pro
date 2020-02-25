QT+= core
QT -= gui


TARGET    = QtCuda
CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any feature of Qt which as been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += main.cpp \


#INCLUDEPATH += \
#       -I/usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc/api/include

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/release/ -lopencv_aruco
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../../usr/local/lib/debug/ -lopencv_aruco
else:unix: LIBS += -L$$PWD/../../../../usr/local/lib/ -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired \
                   -lopencv_calib3d -lopencv_ccalib -lopencv_core -lopencv_cudaarithm -lopencv_cudabgsegm -lopencv_cudacodec -lopencv_cudafeatures2d \
                   -lopencv_cudafilters -lopencv_cudaimgproc -lopencv_cudalegacy -lopencv_cudaobjdetect -lopencv_cudaoptflow \
                   -lopencv_cudastereo -lopencv_cudawarping -lopencv_cudev -lopencv_datasets -lopencv_dnn -lopencv_dnn_objdetect -lopencv_dpm \
                   -lopencv_face -lopencv_features2d -lopencv_flann -lopencv_freetype -lopencv_fuzzy -lopencv_gapi -lopencv_hfs -lopencv_highgui \
                   -lopencv_imgcodecs -lopencv_img_hash -lopencv_imgproc -lopencv_line_descriptor -lopencv_ml -lopencv_objdetect -lopencv_optflow \
                   -lopencv_phase_unwrapping -lopencv_photo -lopencv_plot -lopencv_quality -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape \
                   -lopencv_stereo -lopencv_stitching -lopencv_structured_light -lopencv_superres -lopencv_surface_matching -lopencv_text -lopencv_tracking \
                   -lopencv_video -lopencv_videoio -lopencv_videostab -lopencv_xfeatures2d -lopencv_ximgproc -lopencv_xobjdetect -lopencv_xphoto

INCLUDEPATH += $$PWD/../../../../usr/local/include/opencv2
DEPENDPATH += $$PWD/../../../../usr/local/include/opencv2


LIBS += \
        -L//usr/local/lib/python3.6/dist-packages/torch/lib

win32:CONFIG(release, debug|release): LIBS += -L$$PWD/../../../usr/local/lib/python3.6/dist-packages/torch/lib/release/ -lc10_cuda
else:win32:CONFIG(debug, debug|release): LIBS += -L$$PWD/../../../usr/local/lib/python3.6/dist-packages/torch/lib/debug/ -lc10_cuda
else:unix: LIBS += -L$$PWD/../../../usr/local/lib/python3.6/dist-packages/torch/lib/ -lc10 -lc10_cuda -lcaffe2 -lcaffe2_detectron_ops_gpu \
        -lcaffe2_gpu -lcaffe2_module_test_dynamic -lcaffe2_observers -lfoxi -lfoxi_dummy -lonnxifi -lonnxifi_dummy \
        -lshm -lthnvrtc -ltorch -ltorch_python

INCLUDEPATH += $$PWD/../../../usr/local/lib/python3.6/dist-packages/torch/include
DEPENDPATH += $$PWD/../../../usr/local/lib/python3.6/dist-packages/torch/include


INCLUDEPATH += $$PWD/../../../usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc/api/include
DEPENDPATH += $$PWD/../../../usr/local/lib/python3.6/dist-packages/torch/include/torch/csrc/api/include



INCLUDEPATH += $$PWD/../../../usr/include/python3.6
DEPENDPATH += $$PWD/../../../usr/include/python3.6

INCLUDEPATH += $$PWD/../../../usr/local/cuda/include
DEPENDPATH += $$PWD/../../../usr/local/cuda/include

########################cuda
CUDA_SOURCES += Correlation_Module/correlation_cuda_kernel.cu
# Path to cuda toolkit install
CUDA_DIR = /usr/local/cuda
INCLUDEPATH += $$CUDA_DIR/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
# GPU architecture
CUDA_ARCH = sm_60
# NVCC flags
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v
# Path to libraries
LIBS += -lcudart -lcuda
# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,' -I','-I',' ')
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -O3 -arch=$$CUDA_ARCH -c $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependcy_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -O3 -M $$CUDA_INC $$NVCCFLAGS      ${QMAKE_FILE_NAME}

cuda.input = CUDA_SOURCES
cuda.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}_cuda.o
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_COMPILERS += cuda


