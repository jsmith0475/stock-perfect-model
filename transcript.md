WEBVTT


00:00:00.100 --> 00:00:02.260
I built a trading algorithm using algebraic topology.

00:00:02.261 --> 00:00:04.501
It was consistently profitable for the last five years,

00:00:04.502 --> 00:00:06.301
even when the market was bleeding in 2022.

00:00:06.302 --> 00:00:06.861
The way it works

00:00:06.862 --> 00:00:08.901
is that I modeled the market as a weighted graph of stocks,

00:00:08.902 --> 00:00:10.701
ran a laplacian diffusion on recent returns,

00:00:10.702 --> 00:00:12.741
and took residuals to estimate local mispricings.

00:00:12.742 --> 00:00:14.781
Then I used persistent homology on the correlation structure

00:00:14.782 --> 00:00:16.101
to compress the global market shape,

00:00:16.102 --> 00:00:17.141
and the regime features

00:00:17.142 --> 00:00:19.141
fed these into a proprietary market neutral model.

00:00:19.142 --> 00:00:20.781
I built this to complement a neural net strat 

00:00:20.782 --> 00:00:21.821
I created about a month ago.

00:00:21.822 --> 00:00:24.541
I think it'll be cool to combine these into a kind of ensemble,

00:00:24.542 --> 00:00:25.812
but that's for another video.
