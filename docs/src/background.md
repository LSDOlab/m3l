---
title: Background
---

![alt text](/src/images/SISR.png "SISR diagram")

M3L works together with CADDEE to facilitate solver-independent data transfer. 

Solver independence is achieved by coupling each model with a neuteral framework representation of relevant states.
Data is transfered between model and framework using an intermediate nodal representation.
Models are responsible for generating linear maps between the nodal representation and their internal repersentation of states.
:math: P_{mm'}