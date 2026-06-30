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

性能与效果优化

参考文档：https://xcnuqiaojwqi.feishu.cn/wiki/JZnSw2Lc6iwqsLktl42cjLRmnOf


网络延迟分析
```
python -m quest_hand.plot_teleop_vr_records --network --raw records\test03_vr_raw.csv
```

绘制轨迹，对比 raw_target 和 sent_target：
```
python -m quest_hand.plot_teleop_vr_records records\test03_converted.csv --save records\test03_traj.png --no-show
```

重新计算平滑
```
python -m quest_hand.smooth_trajectory_experiment records\test03_converted.csv --source converted --postprocess-config quest_hand\postprocess_config.yaml --save records\test03_smooth_exp.png --no-show
```


```
python -m quest_hand.plot_teleop_vr_records --network --raw records\test02_vr_raw.csv                                              
python -m quest_hand.smooth_trajectory_experiment records\test02_converted.csv --source converted                                  
python -m quest_hand.plot_teleop_vr_records --robot-state records\test02_robot_state.csv --save records\test02_robot_state_check.png --no-show
```

```
python .\teleop_vr_send.py X5 can0 X5 can1 --record-prefix test_dual --postprocess-config quest_hand\postprocess_config.yaml
```
