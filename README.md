# Project_covid
## University of Glasgow - Online MSc on Data Analytics 
#### Project for Data Programming in Python

This is the end project for the course of Data Programming in Python
##### Project pdf
[a link](https://github.com/gpeddev/Project_covid/blob/master/project.pdf)

##### Initialization day=0
```
s=Settings(m=40,n=25,r=2,k=4,N=200)
sim=Simulation(s)
sim.run()
sim.plot_state(0)
```
[Initialization](https://github.com/gpeddev/Project_covid/blob/master/Figure_1.png)

##### day=101
```
sim.plot_state(100)
```
[day 101](https://github.com/gpeddev/Project_covid/blob/master/Figure_2.png)

##### Chart
```
sim.chart()
```
[chart](https://github.com/gpeddev/Project_covid/blob/master/Figure_3.png)

##### Average chart
```
Simulation.averaged_chart(s,200)
```
[Average chart](https://github.com/gpeddev/Project_covid/blob/master/Figure_4.png)
