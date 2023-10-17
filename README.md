# Factors-that-Lead-to-Incorrectly-Called-Strikes

<figure>
    <p align="center">
    <img src="https://camo.githubusercontent.com/f3172a77fb4aa7991091f53709da677b84bd80dd602e6c80d2f5faef97340afd/68747470733a2f2f7777772e6a736f6e6c696e652e636f6d2f6763646e2f70726573746f2f323032322f30342f32352f555341542f65376135636361362d393865302d343738632d386538662d6661323164626434363536632d41505f427265776572735f5068696c6c6965735f4261736562616c6c2e6a70673f63726f703d323934362c313932312c783236342c79313032"
         alt="Content vs Collab"
         >
    </p>
</figure>

### by John Nahra

# Overview

As a baseball player and Cleveland Guardians fan for close to three decades, I have seen how umpire calls can have a massive impact on the outcome. In this project, I performed two analyses. In the first analysis, I sought to estimate how large an effect missed ball & strike calls had on MLB run production over the course of the 2023 season. On the whole, I found that missed umpire calls on net reduced league run expectancy by almost 200 runs (through August 14). I created a dashboard that highlights the players and teams most affected by missed umpire calls, which can be viewed [here](https://public.tableau.com/app/profile/john.nahra/viz/MLB2023ChangeinRunExpectancyfromMissedBallStrikeCalls/MLB2023ChangeinRunExpectancyfromMissedBallStrikeCallsforTeams). In the second analysis, I ran several machine learning models to better understand the top causes of incorrectly called strikes, utilizing data on pitch characteristics, pitcher & batter characteristics, and game conditions. My best models had almost 90% recall and close to 30% precision when predicting incorrectly called strikes (versus true balls). Some of the top factors were a batter's height, pitch location, and the count. Greater awareness of the factors that lead to incorrectly called strikes can help hitters know when they need to expand the zone, help pitchers take advantage of incorrect strike calls, and potentially help umpires counter their biases.

The pre-Tableau portion of the first analysis can be found in ['Tableau prep-MLB delta run expectancy by base-out-count-pitch'](https://github.com/jnahra/Factors-that-Lead-to-Incorrectly-Called-Strikes/blob/main/Tableau%20prep-MLB%20delta%20run%20expectancy%20by%20base-out-count-pitch.ipynb).

My full second analysis can be found in the ['final_notebook-Factors That Lead to Incorrectly Called Strikes'](https://github.com/jnahra/Factors-that-Lead-to-Incorrectly-Called-Strikes/blob/main/final_notebook-Factors%20That%20Lead%20to%20Incorrectly%20Called%20Strikes.ipynb).

# First Analysis Summary

I used pitch-level data from the 2023 MLB season through August 14 downloaded from Statcast. Only pitches that resulted in a called strike or called ball with no ancillary event were used (i.e. no foul balls, swinging strikes, balls put in play, caught stealing). I wanted to isolate events that were solely affected by the home plate umpire's call. To determine whether an umpire missed the call, I used Statcast's "zone". If Statcast determined that the pitch was outside of the strike zone but the umpire called it a strike, that was a false called strike, whereas a pitch inside the strike zone but called a ball was a false called ball.

To figure out how these missed calls impacted run expectancy, I utilized Statcast's "change in run expectancy" metric. Statcast has a calculated change in run expectancy based on the result of each pitch (and any events that occur, such as a home run or a double play). For my analysis, I only included called balls and strikes. I determined that Statcast has a fixed run expectancy matrix it uses based on the runners on base, number of outs, the count, and the pitch result. Thus, I was able to determine the exact change in run expectancy for each scenario, such as runners on 2nd and 3rd, 1 out, 2-1 count, and a called strike. I also knew what the change in run expectancy would be for that exact scenario except that the pitch result was a called ball. Therefore, I could isolate the effect of missed umpire calls by calculating the total swing in the change in run expectancy from a ball to a strike (and vice-versa) depending on the missed call. False called balls had a positive effect on the change in run expectancy while false called strikes had a negative impact. I created different aggregations of their net effect for the total MLB, for each team, and for each batter in my Tableau dashboard. My Cleveland Guardians were one of the more fortunate teams in 2023, with umpires helping to produce a net positive change in run expectancy from missed calls, one of the few MLB teams to be in the green.

The rest of this README is dedicated to the second analysis.

# Data Understanding and Filtering

I used pitch-level data from the 2023 MLB season through September 25 downloaded from Statcast. Only pitches that resulted in a called strike or called ball were used (i.e. no foul balls, swinging strikes, balls put in play). Since I'm not focused on run expectancy in this analysis, pitches with events such as caught stealing were included. Moreover, this analysis focused solely on false called strikes. Thus, I filtered my data to only include false called strikes and true balls. To determine whether an umpire missed the call, I used Statcast's "zone". If Statcast determined that the pitch was outside of the strike zone but the umpire called it a strike, that was defined as a false called strike, whereas a true ball was a pitch outside of the strike that the umpire correctly called a ball. I also used biographical data on players from Lahmann, including their birth date, debut date, height, weight, whether they have ever been an all-star, etc.

I excluded pitches that had a missing value for "zone", which would remove the ability to determine whether the umpire missed a call, or a missing value for "effective speed", which was one of the independent variables in my models. I looked at the location distribution of the pitches and filtered to only include pitches as far out on the x and z axes as pitches that had been false called strikes. I did this to only include true balls that had some documented potential of being a false called strike. After noticing some rather egregious false called strikes that were outliers, I ultimately went a step further, looking at high, low, inside, and outside false called strikes separately, calculating 1.5*IQR for each, then removing all pitches beyond those measures. In this way, I would be focusing primarily on pitches near the zone that were within the realm of possibility to be either a true ball or false called strike.

The chart below is to give a visual sense of the data points in 2-D space. It shows the four different kinds of pitch results (before I filtered for only false called strikes and true balls). The x and y axis are the horizontal and vertical coordinates of the strike zone, respectively. At this stage, I had filtered for pitches only as far out as false called strikes but had not yet applied outlier trimming. You can see why outlier trimming made sense, as false called strikes were sparse at the outer edges.

<img width="592" alt="image" src="https://github.com/jnahra/Factors-that-Lead-to-Incorrectly-Called-Strikes/assets/122231470/8c7604ad-66f9-4f3f-84ea-6b054612e6e2">


I tried many different variables, different iterations of the same variable, and different binning options before landing on a final set of independent variables, with the intent being to maximize explanatory power, model simplicity, and interpretability of the logistic regression coefficients. The most important factors were as follows:

1. **euclid_dist:** euclidean distance outside of the strike zone. I took the plate x and z coordinates and turned them into absolute distances outside the strike zone (if the pitch was in the strike zone horizontally or vertically, I set the corresponding coordinate equal to zero). Then I simply took the square root of the sum of squared absolute distances. Generally speaking, the farther outside the zone, the less likely that pitch is to be a false called strike.
2. **height*hi_lo:** a hitter's height multiplied by a binary variable equal to 1 if the pitch was high or low (outside the strike zone) else 0. The thinking being that height may matter only on high or low pitches. Tall hitters should have different strike zones but an umpire's zone may remain more fixed to the average height batter, resulting in false called strikes for low pitches. Moreover, there may still be false called strikes for high pitches given that the umpire more rarely calls pitches behind an especially tall player.
3. **high_or_low_pitch:** 1 if the pitch is high or low (outside of the strike zone) else 0. It appears that umpires are much more likely to make a false called strike on a pitch outside of the zone horizontally than a pitch outside of the zone vertically.
4. **woba_by_count:** the average 2023 MLB woba for each count. Rather than have 12 one-hot-encoded categorical variables, I decided to use woba as a proxy for hitter's count. In this way, it turns count into a pseudo-numeric variable that has perhaps more accurate distances between the categories (i.e. the difference between 0-1 and 0-2 is different than the difference between 3-0 and 3-1). The thinking is that umpires are more apt to help pitchers when they are behind in the count and less apt to help pitchers when they are ahead in the count. Woba effectively captures hitter's counts by reflecting how well hitters do in each count.
5. **effective_speed:** the perceived velocity of the pitch, based on velocity and extension. Based on the relationship, it appears that slower pitches are more likely to be false called strikes, probably because a slower pitch implies a pitch with more spin/movement, which may be harder for an umpire to track where it crossed the plate than a straight faster pitch.

<img width="466" alt="image" src="https://github.com/jnahra/Factors-that-Lead-to-Incorrectly-Called-Strikes/assets/122231470/199379c2-7658-45d6-9eb5-ec1be438c49b">


6. **bats:** whether the batter is batting left-handed or right-handed. It appears that umpires are more biased against right-handed hitters with respect to false called strikes.
7. **on_base_bins:** three categories of runners on base. Includes bases loaded, nobody on, and other as the reference. Umps may be more lax with nobody on, and may be hesitant to hurt the pitcher by walking in a run with the bases loaded.
8. **alt_inning_bins:** 1 if inning is 5th or later else 0. The thinking for the split being that relief pitchers usually come into the game in the 5th inning or later, and for night games it tends to be dark in the second half of games.
9. **run_diff_bins:** 1 if either team leads by 4 or more runs else 0. The thinking being that umpires may call a looser game (i.e. more false called strikes) when the game score is not very close.
10. **pitcher_allstar:** 1 if pitcher has ever been an all-star else 0. The thinking being that a good pitcher may get more favorable calls (i.e. more false called strikes).
11. **batter_allstar:** 1 if batter has ever been an all-star else 0. The thinking being that a good hitter may get more favorable calls (i.e. fewer false called strikes).
12. **time_since_debut:** number of days since batter played first MLB game. The thinking being that a longer-tenured hitter may get more respect and favorable calls (i.e. fewer false called strikes).

<img width="466" alt="image" src="https://github.com/jnahra/Factors-that-Lead-to-Incorrectly-Called-Strikes/assets/122231470/921a36d9-ef89-4291-9b53-3a0adc818a66">


A few notable variable omissions:

I considered including strike zone height in my model as well, but it was fairly correlated with height, and I was more interested in looking at the effect of height on false called strikes. Height is a common, easy-to-interpret fact about a player whereas a player's strike zone height is not very well-known and can actually change pitch to pitch. Moreover, players with the same height can have different strike zone heights due to differences in their batting stance. In theory, umpires should be looking at a player's strike zone (where the letters of their jersey and their knees are) but in reality they may have some implicit bias of seeing the height of the player coming to bat.

<img width="383" alt="image" src="https://github.com/jnahra/Factors-that-Lead-to-Incorrectly-Called-Strikes/assets/122231470/a74ac499-aa30-4410-8584-23fc94829dde">


I also created a variable makeup_call_potential, which returned 1 if there was false called ball earlier in the at bat else 0. I wanted to see if there was evidence that the umpire would make up for a false called ball with a false called strike. I did not see significant evidence in my exploratory analysis, and there was also the issue of imbalanced cases (a pitch with a false called ball earlier in at bat was rare). Thus, I ultimately excluded it from the model.

# Modeling

The dependent variable is binary, 0 if true ball 1 if false called strike. About 9.5% of pitches were false called strikes, meaning an imbalanced dataset. I attempted to rectify this two separate ways: the first through corrective class weightings in the models and the second through an under-sampled train dataset.

In each model, I was focused primarily on recall (correctly identifying most false called strikes), but I ran grid searches to optimize for F-1 Score, as I wanted some focus on precision (i.e. not too many false positives) as well so that my models weren't simply predicting all pitches to be false called strikes.

After filtering, I had 128,148 pitch data points, 102,518 in the train set and 25,630 in the test set (80-20 split).

I one-hot-encoded three variables, outs_when_up (how manys outs there were in the inning), bats, and on_base_bins.

I ran two decision tree models, one with the full original train set (and class_weight = 'balanced' parameter), and one with an under-sampled train set to account for class imbalance.

Next, after standard-scaling the numeric variables, I ran a logistic regression with the full original train set (and class_weight = 'balanced' parameter). I also created a pipeline that standard-scaled and ran a logistic regression model on the under-sampled data for comparison.

The third model type I ran was a random forest. I used only the under-sampled data due to its long training time.

Fourth, I ran a neural network model on the full train data with early stopping criteria if recall did not improve after 10 epochs.

Finally, I tried a simple XGBoost model on the under-sampled data.

# Evaluation

I extracted the logistic regression coefficients using the full train set and converted them into interpretable odds (sorted in descending order):

<img width="217" alt="image" src="https://github.com/jnahra/Factors-that-Lead-to-Incorrectly-Called-Strikes/assets/122231470/9f8e26a3-6e20-4fdc-8d30-6794ac256a0c">

For numeric variables such as euclid_dist, the interpretation is that a 1-standard deviation increase in the euclidean distance of a pitch outside the strike zone is associated with a 97% reduction in the likelihood of that pitch being a false called strike.

For categorical variables such as high_or_low_pitch, the interpretation is that a high or low pitch is 65% less likely to be a false called strike than an inside or outside pitch.

Here are the test precision and recall scores for each model, sorted by F-1 score:

<img width="530" alt="image" src="https://github.com/jnahra/Factors-that-Lead-to-Incorrectly-Called-Strikes/assets/122231470/3d83c5d8-303d-496e-bae2-d2e2feb58e15">

You can see most of the models were fairly similar in terms of precision, recall, and F-1 scores.

I left the threshold at 0.5 for the logistic regression model as I felt it was an acceptable balance of precision and recall (given my focus on recall), but I did look at a precision-recall curve to observe the area under the curve and other combinations of precision and recall at different thresholds, including what threshold would maximize the F-1 score. I looked at train/test precision and recall for different max depths for decision trees/random forests and looked at F-1 scores for different batch sizes for the neural network model.

<img width="624" alt="image" src="https://github.com/jnahra/Factors-that-Lead-to-Incorrectly-Called-Strikes/assets/122231470/c1c5837c-178a-4041-9532-20ef4391f3a6">

You can see the trade-off between precision and recall. The threshold with the highest F-1 score has a recall score of 52% and a precision score of 43%.

# Recommendations & Future Insights

Through this analysis, I learned some significant factors that lead to incorrectly called strikes. My models were fairly good at identifying instances of false called strikes. They were certainly not perfect, but that is to be expected. There are many factors outside the scope of this analysis that lead to incorrectly called strikes, including individual umpire strike zone bias, catcher framing, and simple randomness (an umpire could have blinked!). In a future analysis, I could look into potentially gathering umpire and catcher IDs to see if those factors make a difference. But at the end of the day, any model trying to predict false called strikes won't be fool-proof. I could also expand the analysis by trying to predict false called balls versus true strikes. There I suspect catcher framing/where the catcher is set up could play a larger role.

As a current hitter in a Sunday men's league, I will leave some advice for MLB hitters based on the logistic regression odds table.

1. If you're tall, you need to stay cognizant about the potential for false called strikes on high and low pitches. If the pitch is high or low, a 1-standard deviation increase in height is associated with a 97% greater likelihood of a false called strike.
2. False called strikes are much more likely when hitters are ahead in the count than behind. A 1-standard deviation increase in woba (about .100) is associated with a 42% greater likelihood of a false called strike. For some reference, the woba for a 3-0 count is about 100 points higher than that for a 3-1 count.
3. Don't rely on the umpire when the bases are loaded, swing if it's close. False called strikes are 35% more likely when the bases are loaded versus other on-base combinations (excluding nobody on, which is 12% more likely than the other category).
4. In the second half of the game, expand your zone slightly. Maybe because the umpire is getting dialed into seeing pitches from hard-throwing relievers with wicked breaking balls, or because it's getting dark, but false called strikes are 10% more likely in the 5th inning or later versus the first 4 innings.
5. If your team is up or down by 4 runs or more, expand your zone slightly. False called strikes are 7% more likely when the score isn't as close.
6. If the pitcher is/was an all-star, prepare for them to get the benefit of the doubt. All-star pitchers are 4% more likely to get false called strikes.
7. Putting your time in pays off. If you're a veteran, you can better afford to wait for your pitch. For every 1-standard deviation increase in days since your MLB debut, you're 1.4% less likely to face a false called strike.
8. Hey now, you're an all-star! If you are or have been an all-star yourself, congratulations! You're 8% less likely to have a false called strike against you.
9. If it's fast, let is pass. But if it's slow, you may not be able to just let it go. For every 1-standard deviation increase in effective speed, the pitch is 15% less likely to be a false called strike.
10. Fellow lefties rejoice! Left-handed hitters are 18% less likely to have a false called strike than right-handed hitters.
11. As mentioned, protect the corners. Inside and outside pitchers are more likely to be false called strikes. High and low pitches are 65% less likely to be false called strikes.
12. Finally, and most obviously, the worry of a false called strike fades as the pitch is farther and farther outside the zone. For every 1-standard deviation increase in euclidean distance outside the strike zone, the pitch is 98% less likely to be a false called strike.

Thanks for reading and happy hitting!
