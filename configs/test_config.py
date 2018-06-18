# DAVIS model
model = dict(
    name="MP2S",
    backbone=dict(
        type='sense_resnet',
        bn_param=dict(momentum=0.95),
        bn_training=True,
        layers_stride=[1, 2, 1, 1],
        layers_dilation=[1, 1, 2, 4],
        mult_conv_w=[1, 1],  # lr, decay
        mult_conv_b=[2, 0],  # lr, decay
        mult_bn=[1, 1],  # lr, decay
    ),
    aspp=dict(
        channels=512,
        kernel_size=3,
        dilation_series=[6, 12, 18, 24, 1],
        bn_param=dict(momentum=0.9997),
        bn_training=True,
        mult_conv_w=[1, 1],  # lr, decay
        mult_conv_b=[2, 0],  # lr, decay
        mult_bn=[1, 1],  # lr, decay
    ),
    classifier=dict(
        bn_param=dict(momentum=0.9997),
        bn_training=True,
        num_class=2,
        mult_conv_w=[1, 1],  # lr, decay
        mult_conv_b=[2, 0],  # lr, decay
        mult_bn=[1, 1],  # lr, decay
    ))

weight = 'models/MP2S.pth.tar'
