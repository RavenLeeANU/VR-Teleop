连接:
APK中使用手势操作，指定IP地址.
如果TCP报错看看是不是Video Streamer开了之后PC端的Mujoco没装或者是报错了。


实机控制ARX5，支持无硬件SDK的调试操作
```
python .\teleop_vr_send.py X5 can0 --hand left --damping-protection --log-interval 1 --pos-scale 1 --ori-scale 1
```

仿真调试
```
python .\examples\video\x5_video_host.py --mode rotate --arm left --rotate-angle 0.5 --kinematic 
```

IK测试
```
python .\examples\video\x5_ik_test.py --arm right --directions +x,+y,-y
```