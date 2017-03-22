# 数据说明
## linkVolecity
1. 数据包括 time,velocity,week
2. - time:以0-71代表时间片
   - velocity:在某一时间片中平均速度
   - week: 0-6表示Mon－Sun
3. 排除了国庆和中秋的数据

## splitData
1. 将天气，道路数据绑定到table5中，并进行一些计算，具体看表
2. 按照intersection－tollgate 对数据划分

## weather_figure
splitData中的数据绘图结果，包括对每种天气，包括总体的和按照道路的绘制
## vecolity_figure
linkVolecity中数据的绘制
