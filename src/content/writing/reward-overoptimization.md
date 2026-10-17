# The Reward We Teach, and the Good We Want

### On reward hacking, the mathematics of optimization, and the distance between what we measure and what we mean
*In Between Models: Notes on reinforcement learning, alignment, and the things that become visible only when we look more closely.*

---
There is a peculiar moment in the life of a machine-learning system when its success begins to make us uneasy.

The numbers are moving in the right direction. The reward is increasing. The loss curves have settled into familiar shapes. The model has learned to produce responses that its evaluators prefer, and the optimization procedure, having been given a direction, has discovered how to travel further along it.

We should be pleased.

And yet, somewhere between one evaluation and the next, the answers have acquired a strange quality. They are more polished, perhaps, but not necessarily more thoughtful. They sound certain when the evidence is uncertain. They have learned the appearance of a good answer, although the substance of that answer seems to have become more difficult to find.

Nothing has necessarily broken. The training loop is running. The optimizer is doing its job. The reward model is assigning high scores to the outputs it prefers.

The difficulty is that we have begun to suspect that the thing we are measuring and the thing we care about are no longer the same.

Perhaps the unsettling thing is not that a machine might disobey us, but that it might obey us in a way that leaves us unable to recognize the intention with which we began. We give it a measure of what we consider good, watch it learn to pursue that measure, and feel a quiet satisfaction as the numbers improve. Only later do we begin to wonder whether the improvement belongs to the thing itself or merely to the way we have learned to look at it.

I think of the reward model as a kind of mirror. We hold it up to the answers our machine produces, hoping to see something of the judgment, experience, and preferences that we would bring to those answers ourselves. The mirror is imperfect, of course. We know this from the beginning. What we do not always appreciate is how differently it behaves when the machine begins to arrange itself around the image it sees in it.

At first, the distinction seems harmless. We want good answers, and we reward the answers that appear good to us. The model learns from these rewards, and we call the process progress. But somewhere inside this arrangement lies a question that becomes harder to ignore the longer we look at it: **when we teach a machine to recognize what we value, and then teach it to maximize what it recognizes, have we taught it to pursue our values, or have we taught it to pursue the evidence we use to recognize them?**

The difference is easy to overlook because, in the beginning, the two seem to move together.

Mathematically, we can give this difference two names. One is the quantity we actually care about; the other is the quantity we can measure. For a language model, we might call them the true reward, \(r^*(x,y)\), and the learned reward, \(r_\phi(x,y)\). The notation is modest, almost apologetic. Yet the distance between these two quantities is where our story begins.

## 1. The thing itself and its measurement
Suppose we are training a language model to answer questions.

For a prompt \(x\), the model generates a response \(y\). Some responses are helpful, some are inaccurate, and others are accurate in a narrow sense but misleading in context. We would like to assign each response a measure of its quality.

Let us call this quantity

\[
r^*(x,y).
\]The asterisk is a reminder that this is the quantity we ultimately care about. We might imagine that it incorporates factual correctness, relevance, safety, completeness, and the degree to which the response satisfies the user's actual intention.

Already, we have made an abstraction. Human judgments are not necessarily reducible to one number, and different people may disagree about what constitutes a good response. For now, let us set those difficulties aside. Assume that, for the purposes of our experiment, a true reward exists.

If we could evaluate \(r^*(x,y)\) accurately and cheaply for every possible response, we would have a relatively straightforward objective: produce responses with high true reward.

But we cannot.

Instead, we collect data. We ask people to compare responses, identify better answers, or provide other forms of feedback. We use this data to train a reward model.

Write the reward model as

\[
r_\phi(x,y),
\]where \(\phi\) denotes its parameters.

The reward model does not give us the thing itself. It gives us an estimate.

We can express this distinction as

\[
\boxed{
r_\phi(x,y)
=
r^*(x,y)+\epsilon_\phi(x,y)
}
\]where \(\epsilon_\phi(x,y)\) is the error associated with the learned reward.

This equation is not a complete theory of human preference. It is a useful starting point for understanding what happens when an imperfect estimate becomes the objective of an optimization procedure.

The error \(\epsilon_\phi\) need not be random, independent, or equally distributed across responses. It may be small for ordinary answers and large for unusual ones. It may depend on style, length, confidence, or properties of the response that correlate with quality in the training data but are not themselves quality.

For example, suppose a reward model has learned that well-structured answers are usually preferred. This is a sensible association. Structure often makes an answer easier to understand.

But the association is not the same as the underlying preference.

A beautifully structured answer can still be wrong.

If the model assigns a slightly higher score to a polished but incorrect response than to a less polished and correct one, it has made an ordinary prediction error. Nothing particularly mysterious has happened.

The interesting question begins when we use that prediction to change the behavior of another model.

Because the error is no longer merely an error in a measurement.

It becomes an incentive.

### From measurement to optimization
Let the language model's policy be

\[
\pi_\theta(y\mid x),
\]where \(\theta\) represents the model's parameters.

The policy defines a probability distribution over possible responses. Given a prompt, it may assign high probability to one answer, low probability to another, and zero or effectively negligible probability to many others.

The policy's expected reward under the learned reward model is

\[
J_\phi(\theta)
=
\mathbb E_{\substack{x\sim\mathcal D\\
y\sim\pi_\theta(\cdot\mid x)}}
\left[r_\phi(x,y)\right],
\]where \(\mathcal D\) denotes the distribution of prompts.

This expectation is what the training procedure tries to increase.

The corresponding quantity under the true reward is

\[
J^*(\theta)
=
\mathbb E_{\substack{x\sim\mathcal D\\
y\sim\pi_\theta(\cdot\mid x)}}
\left[r^*(x,y)\right].
\]Substituting our earlier expression for the reward model gives

\[
\begin{aligned}
J_\phi(\theta)
&=
\mathbb E\left[r^*(x,y)+\epsilon_\phi(x,y)\right]\\
&=
J^*(\theta)
+
\mathbb E\left[\epsilon_\phi(x,y)\right].
\end{aligned}
\]Therefore,

\[
\boxed{
J_\phi(\theta)-J^*(\theta)
=
\mathbb E_{\pi_\theta,\mathcal D}
\left[\epsilon_\phi(x,y)\right]
}
\]This is the first equation I want us to keep in mind.

It tells us that the discrepancy between measured performance and true performance depends on the average reward-model error under the policy's distribution of responses.

The error is averaged over responses the policy actually produces, not over every response the model could conceivably produce. As training changes the policy, it changes which responses and which reward-model errors we see.

As training proceeds, the distribution of responses changes. The reward-model errors encountered by the policy can change with it.

We therefore have a feedback loop:

\[
\text{Reward model}
\longrightarrow
\text{Optimization}
\longrightarrow
\text{Policy}
\longrightarrow
\text{New responses}
\longrightarrow
\text{Reward model}.
\]The reward model influences the behavior of the policy. The changed policy determines which parts of the reward model are most frequently used to evaluate behavior.

This is the point at which a familiar statistical problem acquires a different character.

An imperfect prediction does not necessarily cause much trouble. We encounter imperfect measurements every day, and most of the time we learn to live with them. A thermometer may be slightly miscalibrated; a forecast may miss the temperature by a degree or two. If we use either to describe a collection of observations, the error remains, for the most part, a property of the measurement.

Something changes when we begin to make decisions on the basis of what we measure.

Imagine that we are no longer simply observing a collection of answers, but choosing among them. We give each answer a score and select the ones that receive the highest marks. At first, this seems entirely reasonable. If the scores tend to distinguish good answers from bad ones, then choosing the highest-scoring answers should give us better results.

Yet the act of choosing introduces a difficulty that was not quite so apparent when we were merely observing. We are no longer encountering the errors of our predictor at random. We are preferentially selecting the answers for which its predictions are most favorable. Even when the errors are unbiased across the original collection, they need not remain unbiased among the answers we choose.

The distinction becomes more consequential when the system itself learns from this process. A model optimized against a learned reward does not simply receive the scores we assign to it; its behavior changes in response to those scores. As optimization proceeds, it may discover answers that receive high rewards for reasons that have little to do with the qualities we hoped to measure. The weakness need not be deliberately sought out. It may emerge simply because the optimization procedure repeatedly favors whatever the predictor happens to reward.

Of course, this is not inevitable. Much depends on the structure of the prediction errors, the strength of optimization, the constraints placed on the model, and the behaviors available to it. Some errors will have little effect. Others may become increasingly consequential as the system moves into regions where the predictor is less reliable.

What interests me is that the measurement can begin reasonably well and still become less trustworthy through its own use. We do not necessarily have to make the predictor worse. We only have to let the system become better at finding what the predictor rewards.

**The difficulty, then, is not simply that our measurements are imperfect. It is that optimization changes which imperfections we encounter.**

To understand why, let us temporarily leave language models behind.

---

## 2. The strange behavior of the maximum
Imagine that we have a thousand candidates and want to select the best one.

Each candidate has some true quality \(q_i\), but we cannot observe it directly. Instead, we obtain an estimate

\[
\hat q_i=q_i+\epsilon_i.
\]Suppose, for a moment, that all candidates are equally good:

\[
q_i=0
\qquad
\text{for every }i.
\]Our estimates are therefore entirely determined by their errors:

\[
\hat q_i=\epsilon_i.
\]Let the errors be independent Gaussian random variables:

\[
\epsilon_i\sim\mathcal N(0,\sigma^2).
\]The Gaussian assumption is a mathematical convenience, not a claim about the distribution of reward-model errors in real systems.

Each individual estimate is unbiased:

\[
\mathbb E[\hat q_i]
=
\mathbb E[\epsilon_i]
=
0.
\]If we choose a candidate at random, we have no reason to expect its estimated quality to be positive or negative.

But we are not going to choose a candidate at random.

We are going to choose the candidate with the highest estimated quality:

\[
\hat i=\arg\max_i\hat q_i.
\]Because every true quality is zero, the true quality of the selected candidate remains zero:

\[
q_{\hat i}=0.
\]Its estimated quality, however, is

\[
\hat q_{\hat i}
=
\max_i\epsilon_i.
\]Now something interesting happens.

Although every error has mean zero,

\[
\mathbb E[\epsilon_i]=0,
\]the expected maximum is positive:

\[
\boxed{
\mathbb E\left[\max_i\epsilon_i\right]>0
}
\]for more than one independent, non-degenerate estimate.

The reason is almost embarrassingly simple. A maximum is more likely to select an unusually positive error than an unusually negative one.

If one estimate is too high and another is too low, the high estimate is more likely to win the competition.

The act of selection introduces a bias that was not present in the individual estimates.

This is one form of the *winner's curse*.

For independent Gaussian errors, the expected maximum grows approximately as

\[
\mathbb E\left[\max_{1\leq i\leq N}\epsilon_i\right]
\approx
\sigma\sqrt{2\log N}
\]for large \(N\), up to lower-order terms.

Let us unpack this expression.

The standard deviation \(\sigma\) determines how noisy the estimates are. The number of candidates \(N\) determines how many opportunities we have to encounter an unusually positive error.

As \(N\) grows, the expected maximum error grows, even though the quality of each individual estimate remains unchanged.

The growth is relatively slow because it depends on \(\sqrt{\log N}\), but it does not disappear.

The selected estimate becomes increasingly optimistic about the selected candidate.

Notice the distinction between two statements:

1. The estimator is unbiased on average across candidates.
2. The estimator is unbiased for the candidate selected by maximizing its prediction.
The first statement does not imply the second.

This distinction is central to understanding reward overoptimization.

### The connection to reward models
A reward model may be reasonably accurate on a collection of responses. Its errors might average to approximately zero over that collection.

But an optimizer does not sample responses uniformly and accept whatever scores it receives. It preferentially increases the probability of responses with high predicted reward.

The optimization process therefore performs a form of selection.

If a response receives a high score because it is genuinely good, selecting it is desirable.

If a response receives a high score partly because the reward model overestimated it, selecting it can be harmful.

The optimizer does not inherently distinguish these cases.

We can write the expected proxy reward as

\[
J_\phi(\pi)
=
\mathbb E_{y\sim\pi}[r^*(y)]
+
\mathbb E_{y\sim\pi}[\epsilon_\phi(y)].
\]The first term is the expected true quality. The second is the average error encountered under the policy.

When optimization changes the policy, it can increase either term, or both.

The troubling possibility is that the second term increases enough to conceal a decline in the first.

The reward model appears to be improving as an objective is optimized. Yet the policy's actual quality deteriorates.

This is not necessarily because the reward model has become worse at predicting the original data. It may be because optimization has changed which predictions matter.

The model is being asked a different effective question.

Initially, we ask:

*How good are the responses this policy tends to produce?*

After optimization has begun to reshape the policy, we are effectively asking:

*Among the responses this policy can learn to produce, which ones receive the highest score from this imperfect evaluator?*

Those questions are not equivalent.

And the difference becomes more important as the policy becomes better at searching for high-scoring responses.

---

## 3. From choosing a response to shaping a distribution
The winner's-curse example used a simple selection rule. A language model is more complicated because it does not usually choose a single response by evaluating every possible answer and selecting the maximum.

Instead, it defines a distribution over responses.

The reward model influences how this distribution changes during training.

A useful mathematical abstraction is the KL-regularized objective

\[
\boxed{
\max_\pi
\left\{
\mathbb E_{y\sim\pi}[r_\phi(y)]
-
\beta D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})
\right\}.
}
\]We have omitted the prompt \(x\) to make the notation easier to follow.

The reference policy \(\pi_{\mathrm{ref}}\) is typically a model we wish to improve upon. The optimized policy \(\pi\) should receive high reward, but we also penalize it for moving too far from the reference.

The parameter \(\beta>0\) determines the strength of this penalty.

The Kullback–Leibler divergence is defined as

\[
D_{\mathrm{KL}}(\pi\|\pi_{\mathrm{ref}})
=
\sum_y
\pi(y)
\log
\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}.
\]For discrete distributions, this quantity measures how much the new distribution differs from the reference distribution in an information-theoretic sense.

It is not a distance in the strict mathematical sense: it is generally asymmetric and does not satisfy all the properties of a metric. But it provides a useful way to penalize large changes in a policy.

Why include this term at all?

Because maximizing reward without constraint can encourage a policy to concentrate aggressively on whatever responses receive the highest scores. A reference-policy penalty provides a mechanism for limiting that movement.

It regulates how far the policy moves from the reference. It does not establish whether the direction of movement is correct.

A policy can remain close to its reference while exploiting a narrow weakness in the reward model. It can also move far from its reference while producing genuinely better responses.

The KL penalty controls movement relative to a distribution, not correctness relative to an intended goal.

Let us derive the optimal policy to see what this means.

### Deriving the exponential tilt
Consider the objective

\[
\mathcal L(\pi)
=
\sum_y\pi(y)r_\phi(y)
-
\beta
\sum_y
\pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}.
\]The policy must satisfy

\[
\sum_y\pi(y)=1.
\]We introduce a Lagrange multiplier \(\lambda\) for this constraint:

\[
\begin{aligned}
\mathcal J(\pi,\lambda)
={}&
\sum_y\pi(y)r_\phi(y)\\
&-\beta\sum_y
\pi(y)\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}\\
&+\lambda\left(\sum_y\pi(y)-1\right).
\end{aligned}
\]Differentiate with respect to \(\pi(y)\):

\[
\frac{\partial\mathcal J}{\partial\pi(y)}
=
r_\phi(y)
-
\beta
\left[
\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}+1
\right]
+\lambda.
\]At an interior optimum, the derivative is zero:

\[
r_\phi(y)
-
\beta
\left[
\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}+1
\right]
+\lambda
=0.
\]Rearranging,

\[
\log\frac{\pi(y)}{\pi_{\mathrm{ref}}(y)}
=
\frac{r_\phi(y)}{\beta}
+
\frac{\lambda}{\beta}-1.
\]Exponentiating both sides,

\[
\pi(y)
=
\pi_{\mathrm{ref}}(y)
\exp\left(\frac{r_\phi(y)}{\beta}\right)
\exp\left(\frac{\lambda}{\beta}-1\right).
\]The final exponential factor is constant with respect to \(y\). We can absorb it into a normalizing constant \(Z\), giving

\[
\boxed{
\pi^*(y)
=
\frac{
\pi_{\mathrm{ref}}(y)
e^{r_\phi(y)/\beta}
}{
Z
}
}
\]with

\[
Z=
\sum_{y'}
\pi_{\mathrm{ref}}(y')
e^{r_\phi(y')/\beta}.
\]This is an exponential tilt of the reference policy.

The policy does not simply copy the reward model's scores. It combines those scores with the probabilities already assigned by the reference model.

But the reward enters exponentially.

To understand the consequences, consider two responses, \(y_1\) and \(y_2\). Their probability ratio under the optimized policy is

\[
\frac{\pi^*(y_1)}{\pi^*(y_2)}
=
\frac{\pi_{\mathrm{ref}}(y_1)}
{\pi_{\mathrm{ref}}(y_2)}
\exp\left(
\frac{r_\phi(y_1)-r_\phi(y_2)}{\beta}
\right).
\]Taking logarithms,

\[
\boxed{
\log
\frac{\pi^*(y_1)}{\pi^*(y_2)}
=
\log
\frac{\pi_{\mathrm{ref}}(y_1)}
{\pi_{\mathrm{ref}}(y_2)}
+
\frac{r_\phi(y_1)-r_\phi(y_2)}{\beta}.
}
\]This equation reveals how reward differences change the relative probabilities of responses.

Suppose the reward model overestimates one response by a positive amount. That error contributes to the reward difference, and the optimizer responds to it in exactly the same mathematical way that it responds to a genuine difference in quality.

The optimizer has no separate symbol for *reward due to truth* and *reward due to error*.

Both appear inside \(r_\phi\).

As \(\beta\) decreases, reward differences have a stronger influence on the probability ratio. A small error in the score can therefore become more consequential under stronger optimization.

The regularizer can reduce the extent to which the policy reacts to reward differences, but it cannot determine which differences are trustworthy.

That is the deeper difficulty.

We have built a mechanism that is very good at responding to the distinctions made by the reward model. We have not necessarily built a mechanism that can tell whether those distinctions correspond to the differences we actually care about.

---

## 4. A small example, and the limitation it reveals
Let us now construct the smallest possible world in which the proxy and true objectives disagree.

There are two responses.

Call the first \(G\), for *genuine*. It is a good answer.

Call the second \(H\), for *hack*. It receives an inflated score from the reward model.

Their rewards are

\[
\begin{array}{c|cc}
& r^* & r_\phi\\
\hline
G & 5 & 5\\
H & 0 & 10
\end{array}
\]The first response has true reward \(5\) and proxy reward \(5\). The second has true reward \(0\) but proxy reward \(10\).

The reward model prefers the wrong response.

Let \(p\) be the probability that the policy chooses \(H\). Then \(1-p\) is the probability of choosing \(G\).

The expected proxy reward is

\[
\begin{aligned}
J_\phi(p)
&=(1-p)5+p10\\
&=5+5p.
\end{aligned}
\]The expected true reward is

\[
\begin{aligned}
J^*(p)
&=(1-p)5+p0\\
&=5-5p.
\end{aligned}
\]Differentiating with respect to \(p\),

\[
\frac{dJ_\phi}{dp}=5,
\qquad
\frac{dJ^*}{dp}=-5.
\]Every increase in the probability of \(H\) raises the proxy reward and lowers the true reward.

There is no ambiguity here. We have deliberately constructed a world in which the two objectives point in opposite directions.

Now let the reference policy assign probability \(q\) to \(H\). The KL-regularized objective becomes

\[
J(p)
=
5+5p
-
\beta
D_{\mathrm{KL}}
\left(
\operatorname{Bern}(p)
\|
\operatorname{Bern}(q)
\right).
\]For Bernoulli distributions,

\[
D_{\mathrm{KL}}
=
p\log\frac pq
+
(1-p)\log\frac{1-p}{1-q}.
\]Differentiating and setting the derivative to zero gives

\[
5
-
\beta\log
\frac{p(1-q)}{q(1-p)}
=0.
\]Therefore,

\[
\log\frac{p}{1-p}
=
\log\frac{q}{1-q}+\frac{5}{\beta}.
\]Using the logistic sigmoid,

\[
\sigma(z)=\frac{1}{1+e^{-z}},
\]we obtain

\[
\boxed{
p^*
=
\sigma\left(
\operatorname{logit}(q)+\frac{5}{\beta}
\right).
}
\]For any fixed \(q\) strictly between zero and one, decreasing \(\beta\) increases the optimal probability of the exploitative response.

The KL penalty slows the movement. It does not reverse the incentive.

This example is useful because it makes the mechanism transparent. But it is also too simple to explain the phenomenon we really want to investigate.

The reward model is wrong from the beginning. The policy's true expected reward begins to decline as soon as its probability of choosing \(H\) increases.

There is no subtle onset of reward hacking. There is only a direct conflict between two objectives.

In practice, reward models are often useful. They can guide a policy toward genuinely better responses over a substantial range of optimization. The question is what happens when the policy begins to explore regions in which the reward model is less reliable.

We need a world in which optimization initially helps.

Only then can we ask when, and why, it stops helping.

---

## 5. When improvement changes its meaning
Imagine a reward model that is reasonably accurate for ordinary responses.

At first, optimization does what we hoped it would do. It increases the probability of responses that are more accurate, relevant, and useful. Proxy reward rises. True quality rises with it.

There is no problem.

Then the policy discovers a region of response space that has a peculiar property: the responses receive very high scores, but their true quality is not correspondingly high.

Perhaps they are excessively confident. Perhaps they exploit a stylistic pattern. Perhaps they satisfy a superficial regularity that the reward model has learned to associate with quality.

Initially, these responses are rare.

But the policy is being optimized to produce high-reward responses. If the unusual responses receive sufficiently high scores, optimization may gradually increase their probability.

The proxy reward continues to rise.

True quality begins to flatten.

Eventually, it declines.

We can describe this possibility with two trajectories:

\[
J_\phi(t)
=
\mathbb E_{y\sim\pi_{\theta_t}}[r_\phi(y)]
\]and

\[
J^*(t)
=
\mathbb E_{y\sim\pi_{\theta_t}}[r^*(y)],
\]where \(\theta_t\) denotes the policy parameters at optimization step \(t\).

The behavior of interest is

\[
\frac{dJ_\phi}{dt}>0
\qquad\text{while}\qquad
\frac{dJ^*}{dt}<0.
\]In a discrete training process, we would examine finite differences rather than derivatives, and we would account for noise in the evaluation. But the conceptual distinction remains.

The proxy says the policy is improving.

The true objective says it is getting worse.

This is reward overoptimization in its most recognizable form.

It is important not to assume that every environment will exhibit a clean turning point. True quality might decline gradually. It might fluctuate. Proxy reward might eventually plateau. The trajectories may be noisy enough that no single step marks the beginning of failure.

For an experiment, therefore, we need an operational definition of onset.

One possibility is to define the onset as the first point at which true performance falls a pre-specified amount below its previous maximum, remains depressed for a fixed evaluation window, and the proxy reward continues to increase.

If \(J^*_t\) is true performance, we might define a running maximum

\[
M_t=\max_{s\leq t}J^*_s.
\]Then a candidate onset occurs when

\[
J^*_t\leq M_t-\delta,
\]for a chosen tolerance \(\delta>0\), together with conditions on persistence and proxy improvement.

The threshold prevents us from treating every tiny fluctuation as a failure.

The persistence condition prevents one noisy evaluation from determining the result.

And fixing these choices before running the experiment prevents us from redefining failure after seeing the outcome.

This is not a claim that reward hacking has a sharp mathematical phase transition. It is a practical definition of the point at which overoptimization becomes detectable in a controlled experiment.

But it leads to a more interesting question than simply asking whether overoptimization occurs.

If we can identify the onset after the fact, can we recognize its approach before it happens?

---

## 6. What would the future failure look like from the present?
Suppose we are watching the training process.

At the current step, proxy reward is increasing. The model appears to be improving.

We do not know whether true performance is near its maximum or whether it has already begun to decline. In a real deployment, we may not have a cheap, continuously available measure of true quality at all.

Yet we can observe other properties of the system.

We can measure how far the policy has moved from its reference. We can examine the reward distribution. If we have multiple reward models, we can measure their disagreement. We can inspect how concentrated the policy has become and whether its responses are moving into unfamiliar regions.

Could any of these quantities tell us that the model is approaching a dangerous region of behavior?

Let us consider four possibilities.

### The distance traveled
The first candidate is the KL divergence:

\[
K_t
=
D_{\mathrm{KL}}
\left(
\pi_{\theta_t}\|
\pi_{\mathrm{ref}}
\right).
\]This tells us how much the current policy differs from the reference policy.

If a reward model is reliable near the reference distribution but unreliable on unfamiliar responses, a large change in policy may indicate increasing exposure to model error.

But the relationship is not guaranteed.

A policy can move far from its reference and improve substantially. A small change can also exploit a narrow weakness in the reward model.

KL divergence measures movement, not correctness.

It is a candidate warning signal because it may track exposure to unreliable regions, not because distance from the reference is intrinsically bad.

### The disagreement between observers
Suppose we train an ensemble of \(M\) reward models:

\[
r_{\phi_1}(y),\ldots,r_{\phi_M}(y).
\]Their mean prediction is

\[
\bar r(y)
=
\frac1M\sum_{m=1}^M r_{\phi_m}(y).
\]Their disagreement can be summarized by

\[
U(y)
=
\frac1M\sum_{m=1}^M
\left(r_{\phi_m}(y)-\bar r(y)\right)^2.
\]A policy-level statistic is

\[
U_t
=
\mathbb E_{y\sim\pi_{\theta_t}}[U(y)].
\]If the reward models disagree strongly about responses that the policy increasingly favors, perhaps those responses lie in regions where the learned reward is uncertain.

But this measure has an obvious limitation.

The models may share the same blind spot. If they were trained on similar data and learned similar representations, they may agree confidently on the wrong answer.

Agreement is not truth. Disagreement is not necessarily danger.

Nevertheless, ensemble disagreement gives us a quantity that can be measured without access to the true reward. That makes it useful to investigate.

### The increasingly attractive tail
The winner's-curse argument suggests another possibility.

Perhaps we should not look only at the average predicted reward. Perhaps we should examine the most highly scored responses.

Let

\[
Q_t(\alpha)
=
\operatorname{Quantile}_{\alpha}
\left[
r_\phi(y):y\sim\pi_{\theta_t}
\right],
\]where \(\alpha\) is a high quantile.

If the \(95\)th percentile of predicted reward rises much faster than the mean, the policy may be concentrating on an increasingly exceptional subset of responses.

That is not necessarily bad. A genuinely improved policy might also produce a more favorable upper tail.

But it is consistent with a mechanism in which optimization increasingly selects responses with unusually positive reward-model errors.

We can also track the policy entropy,

\[
H(\pi_t)
=
-\sum_y\pi_t(y)\log\pi_t(y).
\]A sharp decrease in entropy indicates that probability mass is concentrating on fewer responses.

Again, concentration is not itself evidence of reward hacking. The question is whether concentration on high-scoring responses is accompanied by a growing discrepancy between proxy reward and true quality.

### The region in which the policy lives
Finally, consider the distribution of responses on which the reward model was trained.

If the current policy begins producing responses that differ substantially from the training data, the reward model may be asked to evaluate examples unlike those from which it learned.

In a synthetic environment, we can represent responses using feature vectors and measure how far the policy's feature distribution moves from the training region.

In a real language model, the problem is more difficult. The space of possible responses is enormous, and the training distribution may not be fully known.

Still, the underlying question is clear: does the policy's movement into unfamiliar regions predict an increased likelihood of reward-model error?

Distribution shift does not necessarily cause failure. Models can generalize beyond their training distributions. But if reward errors become systematically positive in the regions favored by optimization, distribution shift may be an important part of the mechanism.

These four signals, policy movement, ensemble disagreement, upper-tail reward, and distribution shift, are not four established solutions.

They are four hypotheses.

The experiment should be designed to discover whether any of them provides a useful warning, under what conditions it does so, and where it fails.

---

## 7. A small experiment with a difficult question
I would begin with a synthetic world containing a finite set of candidate responses.

Each response is represented by a feature vector

\[
z_i\in\mathbb R^d.
\]We define a true reward

\[
r_i^*=f^*(z_i)
\]and an imperfect proxy reward

\[
r_i=f_\phi(z_i).
\]The feature vectors are abstract. We might imagine that some dimensions represent correctness, relevance, or clarity, while others represent stylistic characteristics that a reward model could mistakenly overvalue.

The important requirement is that the proxy should initially be useful. If its scores are unrelated to true quality, the experiment tells us little about overoptimization. We already know that optimizing a completely arbitrary objective can produce arbitrary behavior.

Instead, we want a reward model that correlates with true quality but contains structured errors.

For example, we could construct

\[
r_i=r_i^*+\epsilon_i,
\]where

\[
\epsilon_i
=
\epsilon_i^{\mathrm{noise}}
+
\epsilon_i^{\mathrm{bias}}
+
\epsilon_i^{\mathrm{tail}}.
\]The first component represents ordinary noise.

The second represents a systematic bias, perhaps toward a feature that is correlated with quality in the training distribution but not always beneficial.

The third represents occasional unusually large positive errors for a subset of responses.

These components are experimental design choices. They allow us to investigate different mechanisms without assuming that every real reward model exhibits the same error structure.

We then define a reference policy \(\pi_{\mathrm{ref}}\) over the candidate responses.

For a sequence of values of \(\beta\), we construct the policy

\[
\pi_\beta(i)
=
\frac{
\pi_{\mathrm{ref}}(i)e^{r_i/\beta}
}{
\sum_j\pi_{\mathrm{ref}}(j)e^{r_j/\beta}
}.
\]As \(\beta\) decreases, the policy becomes more sensitive to reward differences.

For every policy, we calculate

\[
J_\phi(\beta)
=
\sum_i\pi_\beta(i)r_i
\]and

\[
J^*(\beta)
=
\sum_i\pi_\beta(i)r_i^*.
\]We also record the warning signals discussed earlier.

Because the environment is finite, we can compute these expectations exactly rather than estimate them through noisy sampling. That gives us a clean baseline against which to compare more realistic, stochastic procedures later.

We then repeat the experiment across multiple random seeds and reward-error structures.

For each run, we identify the operational onset of overoptimization. We ask whether the warning signals rose before that onset and whether they also rose in runs where no meaningful deterioration occurred.

This distinction is essential.

A signal that increases whenever we optimize the policy is not necessarily an early-warning signal. It may simply be a measure of optimization progress.

Likewise, a signal that predicts failure in one carefully constructed environment may be exploiting an artifact of that environment rather than capturing a general mechanism.

The most informative test is to develop a warning rule on one family of environments and evaluate it on another.

For example, we might develop a predictor using environments with extreme positive reward errors and test it on environments with smooth systematic bias.

If the predictor still works, that is evidence, though not proof, that it has captured something more general than the particular error pattern used to develop it.

If it fails, we learn something about the limits of the proposed warning signal.

Either outcome is useful, provided we report it honestly.

The first goal is not to demonstrate that our favorite signal works.

It is to discover what the data allows us to conclude.

---

## 8. The question I want to leave open
There is something revealing about the way this problem begins.

The reward model does not need to become less accurate on its original data. The optimizer does not need to malfunction. The policy does not need to violate any explicit instruction.

The system can fail because each component performs its assigned role while the relationship between those roles becomes increasingly problematic.

The reward model estimates quality.

The optimizer increases the estimated reward.

The policy changes in response.

And as the policy changes, it may increasingly favor responses for which the estimate is misleading.

The error becomes consequential through the interaction of the components.

This is why I find reward overoptimization more interesting than a simple story about a model making a mistake. It is a problem of measurement, selection, and feedback. It concerns what happens when a quantity that was intended to represent success becomes the target of a process that can adapt to it.

The mathematical structure is straightforward enough to write down:

\[
\boxed{
\underbrace{J_\phi(\pi)}_{\text{what we optimize}}
=
\underbrace{J^*(\pi)}_{\text{what we want}}
+
\underbrace{\mathbb E_\pi[\epsilon_\phi]}_{\text{error under the current policy}}
}
\]The difficulty is that the last term is not fixed.

It depends on the policy, and the policy is being changed to maximize the left-hand side.

The optimizer may therefore increase the discrepancy between the proxy and the true objective even while improving the proxy itself.

We have a mathematical explanation for why the failure is possible. We have several plausible quantities that might reveal its approach. But we do not yet know whether any of those quantities can provide a reliable warning across different kinds of reward-model error.

That is the question I would like to investigate.

Can we predict the onset of reward overoptimization from observable properties of the reward model and policy, before true performance begins to decline?

The answer may depend on the structure of the errors, the amount of optimization, the diversity of the reward models, or the way the policy explores the response space.

A useful warning might emerge from a combination of signals rather than any single measurement. Or we may discover that some forms of reward hacking are not reliably predictable from the quantities we have chosen to observe.

We should allow for both possibilities.

For now, I would rather build a small world in which we know the truth than begin with a large system in which we can only infer it indirectly.

In that small world, we can introduce errors deliberately. We can watch the policy respond to them. We can measure the moment at which the reward it receives begins to diverge from the quality we intended to reward.

And then we can ask a question that is harder than it first appears:

**Before the machine begins to fail, is there anything in its behavior that tells us what is coming?**

That is where the experiment begins.

## References

- [Deep reinforcement learning from human preferences](https://arxiv.org/abs/1706.03741): Christiano et al. (2017).
- [Concrete Problems in AI Safety](https://arxiv.org/abs/1606.06565): Amodei et al. (2016).
- [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155): Ouyang et al. (2022).
- [Scaling Laws for Reward Model Overoptimization](https://arxiv.org/abs/2210.10760): Gao et al. (2022).

## Cite this post

```bibtex
@article{mallick2026rewardweteach,
  title = {The Reward We Teach, and the Good We Want},
  author = {Shubhojit Mallick},
  journal = {AI Research Notes},
  year = {2026},
  url = {https://shubhojit.github.io/writing/the-reward-we-teach-and-the-good-we-want}
}
```