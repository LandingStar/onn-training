From the physical structure, it's more like a autoencoder
$$x\stackrel{encoder}{\longrightarrow}z\stackrel{decoder}{\longrightarrow}x'$$
However, we got some perturbation on input $x$ and transmission $z$, so the actually running is like:
$$x\longrightarrow x_m\stackrel{encoder}{\longrightarrow}z_m\stackrel{fibers}{\longrightarrow}z\stackrel{decoder}{\longrightarrow}x'$$
let perturbation on $x,z$ be $\mathcal{X},\mathcal{Z}$ respectively, effective encoder is:
$$E_{eff}:x\mapsto \mathcal Z\circ Encoder\circ\mathcal X(x)$$
where $\mathcal Z\circ Encoder\circ\mathcal X(x)$ is a distribution, fitting the requirement in VAE.

Next we should think how to apply KL div on $x,x'$? Emmm... Maybe applying on $x,x'$ directly like MSE does is acceptable... Let's do like this first. It seems that the KL div in VAE and KL div item in loss aren't the corresponding ones, anyway, please focus on *Supplementary Note 1* in SM of the article to figure it out. We shall use it in the way we've guessed.

From the article we know that they define the loss as:
$$loss=\alpha KL+\beta MSE+ \gamma OP$$
where OP is a penalty term, depicting the sensitivity of model to noise. 

In the article, OP is composed of an $l_1$ or $l_2$ norm between the middle results and the ideal output, which we don't quite understand yet. In other resource, OP in AE can be defined as the summation of derivative pointwisely.

Okay, with the help of AI we confirm that in the def of $l_{OP}$, the output is exactly the input $x$, which force model less affect the overall structure of img. And thus $\gamma$ should be small to guarantee it a soft force. In the similar part of cited article it's 0.3. See more details in the dialogue with [Google Gemini](https://gemini.google.com/app/c447eca58e7ed900?hl=zh-cn)

As the start of simulation, we set layer num 2 and distance 20cm pixel size 15, phase mask neural num 200x200, wave length 1550nm. Also, as general mode, we set $M_0$ a uniform constant, $\alpha=1,\beta=1,\gamma=0.1$

From supplementary materials, we set SNR 24dB(1171km), and phase noise a Gaussian noise with standard deviation $\frac{\pi}{20}$.

We shall follow the structure of previous task. However as non-supervisor training, we need sample creator. And reform the loss calculation. For digital transmission, we got the accuracy judgment, which we'll start from. Also, we need an addition layer to depict fiber coupling.

So we got our tasks:
>1.sample creator (digital) (check)
>1.5. basic net structure (check)
>2.loss calculator (still you can refer to [Google Gemini](https://gemini.google.com/app/c447eca58e7ed900?hl=zh-cn)) (check)
>3.acc calculator
>4.fiber coupling layer(check)

about loss function, we got three item$$loss=\alpha l_{KL}+\beta l_{MSE}+ \gamma l_{OP}$$
where $l_{KL},l_{OP}$ needs middle results, especially $l_{OP}$, so we consider integrating loss calculation into layers, and control them with a global variable: CALCULATE_LOSS. Temporarily, we set no_loss() as the context manager as follow:
```python
from contextlib import contextmanager  
CALCULATE_LOSS=True
@contextmanager  
def no_loss(name):  
    CALCULATE_LOSS=False
    yield  
    CALCULATE_LOSS=True
```
and inside the model (or layer), `CALCULATE_LOSS`controls whether calculate the loss with an `if`.
 All done except KL loss, one thing is how to set the value of $M_0$, yet we don't know the intensity in fibers. So we may deal with fiber coupling first.

About fiber coupling, some info. has been give by the article, however we don't have idea with the fiber's output mode, both intensity and phase.

As the data we've searched, we adapt Daheng Optics DH-FSM1550 series fiber for 1550nm light. Its diameter of mode is $9.5\pm 0.5\mu m$ :[More info.](http://www.cdhcorp.com.cn/gxtxxl/index.htm)
For the lens, article didn't mention their focal length. According to the data in Daheng's shop, we may adapt 15.6mm for 1550nm. [More info.](http://www.cdhcorp.com.cn/gxzzjxl/index.htm)

And about coupling, let the electric field of the fundamental mode on the fiber edge be 
$$E_{ff}=\frac{2}{\sqrt \pi \omega_0}\exp[-(\frac{r}{\omega_0})^2]$$
and the electric field intensity coupled is proportional to
$$\int_{surface}E_{if}E_{ff}^*\text d S$$
where $E_{if}$ is the complex electric field the lens casts on it, which is just the FFT of field on the lens' plate. (~~seem that there's no need of focal length, no wonder that it's not mentioned~~, no, we got distance needed when doing FFT)
We can drop the normalization coefficient temporarily.
After we finish this work with this model, may consider replacing it with FDTD result.

And about the output, as single mode fibers, the only mode has been shown above, only to multiply complex intensity $E_i$.

And about KL loss, the object being estimated is the intensity $I=\left\vert E\right\vert^2$, so $M_0$ should be a non-negative real number.

Now our to-do list remains only the acc judgment. Tomorrow, we'll wind it up and start to construction training process. First training run will start from afternoon if all goes will.

All right, bad news, we got our alarm failed this morning, so it comes 13:35 now.

Now we are arriving at the acc calculation, this is specially for digital. We would take the binary and compare with original message, which require data generator also gives out the sequence.

Emm... fine maybe it doesn't deserve as an individual function. Now we check the training process. 
Wait, maybe we should pack the generator into an iterable item.

Done, move to training.

Bad news, it runs, but no improve with the model, that is, it still output some noise...

Anyway, suggesting check layer by layer