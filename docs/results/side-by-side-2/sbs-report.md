| model | pair | n | both silent | one silent | exact | exact rate | median similarity | similarity >= 0.8 |
|---|---|---|---|---|---|---|---|---|
| baseline-e2 | typed vs A | 65 | 0 | 0 | 0 | 0.00 | 0.48 | 6 |
| baseline-e2 | A vs B | 65 | 0 | 0 | 1 | 0.02 | 0.50 | 3 |
| baseline-e2 | typed vs B | 65 | 0 | 0 | 0 | 0.00 | 0.49 | 5 |
| cons1-e2 | typed vs A | 65 | 2 | 11 | 2 | 0.03 | 0.54 | 11 |
| cons1-e2 | A vs B | 65 | 10 | 3 | 0 | 0.00 | 0.58 | 7 |
| cons1-e2 | typed vs B | 65 | 1 | 10 | 0 | 0.00 | 0.50 | 4 |

| model | rendering | n | silent | onset 0 | onset 1 | onset 2 | onset 3 | onset other | median words | repetitive | unfinished |
|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline-e2 | typed | 65 | 0 | 0 | 0 | 0 | 65 | 0 | 66 | 3 | 17 |
| baseline-e2 | A | 65 | 0 | 52 | 0 | 13 | 0 | 0 | 67 | 6 | 16 |
| baseline-e2 | B | 65 | 0 | 52 | 0 | 12 | 1 | 0 | 66 | 5 | 13 |
| baseline-e2 | A-swap | 26 | 0 | 26 | 0 | 0 | 0 | 0 | 61 | 0 | 1 |
| baseline-e2 | typed-swap | 26 | 0 | 0 | 0 | 0 | 26 | 0 | 67 | 1 | 6 |
| cons1-e2 | typed | 65 | 2 | 0 | 0 | 0 | 49 | 14 | 64 | 7 | 11 |
| cons1-e2 | A | 65 | 13 | 52 | 0 | 0 | 0 | 0 | 61 | 7 | 6 |
| cons1-e2 | B | 65 | 10 | 52 | 0 | 0 | 3 | 0 | 62 | 5 | 7 |
| cons1-e2 | A-swap | 26 | 0 | 26 | 0 | 0 | 0 | 0 | 59 | 2 | 2 |
| cons1-e2 | typed-swap | 26 | 0 | 0 | 0 | 0 | 24 | 2 | 59.5 | 3 | 0 |

| model | pair | both spoke | same first word |
|---|---|---|---|
| baseline-e2 | typed vs A | 65 | 48 |
| baseline-e2 | A vs B | 65 | 52 |
| baseline-e2 | typed vs B | 65 | 43 |
| cons1-e2 | typed vs A | 52 | 44 |
| cons1-e2 | A vs B | 52 | 46 |
| cons1-e2 | typed vs B | 54 | 48 |

### dlg-00005-010 turn 0

**Pupil:** so, um, what actually is a derivative?

**Tape tutor:** fair question, and I'd rather find it from numbers, because a slope only shows itself when a height changes. I expect this cubic to climb to a hump, fall into a dip, and then climb again. at two, I cube it for eight, take away three twos, and so I hold that the curve stands at a height of two there. if the input moves from two to two point one, how much will the height grow, do you think?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | a derivative is a number that tells you how fast something changes, so let's work one out. we should find this cubic curve crossing the middle at zero, since it has to drop below zero in the middle. at zero, zero times zero is zero, and one times zero is zero, so we have the curve sitting at height zero. if we nudge the input to one tenth, how many times that step will the height drop? | (silence) |
| A | a | (silence) |
| B | a | (silence) |

### dlg-00005-010 turn 2

**Pupil:** zero point nine six, because the curve climbs about nine per step.

**Tape tutor:** right, nine and a bit. at two point one, I cube it for nine point two six one, less six point three, so the curve sits at two point nine six one. that rise is about nine times the step. shrink the step, and I'd say the curve at two climbs nine for every one across. so the slope there is nine. at minus two, where the height is minus two, how steep will it be?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, about nine per step, so I hold the curve at two point one stands just under nine tenths high. nine tenths is nine point nine six, so I hold the curve barely climbs past its hump. the slope is nine point nine six, so the curve climbs nearly nine per step at two. at zero, where the height is zero, how much will the slope read? | yes, about nine per step, and the numbers back that up. if the input moves from two to two point one, I hold the curve climbs about nine tenths in one tenth along. the slope is nine, so the curve climbs nine per step at two. at zero, where the curve is flat, how much will the slope read? |
| A | right, nine tenths, so I hold the curve at two point one climbs about nine tenths for each one across. the slope is nine, so the curve climbs nine for each one across there, just as you said. at the top of the hump, at one point five seven, how much will the slope read? | yes, about nine tenths, and that number is the one we should test on the hill. if the hill were flat, I'd hold that a tenth of step gave a tenth of height. the slope is nine, so the hill climbs nine per step along there. at zero, where the curve is flat, how much will the slope read? |
| B | right, zero point nine six, and that number is the one we should find at two point one. I hold that the curve barely climbs there, since two point one cubed over eight is only a tenth over three. so the height grew only a tenth in a tenth of a step, and I take it the curve is level at two. the slope is zero, so the curve is flat there, though its height is two. at zero, where the height is zero, how much will the slope read? | yes, zero point nine six, and that number is the one we should test at two point one. if the curve climbs nine tenths per step at two, I hold it must climb nine tenths per step at two point one. the slope is nine, so the curve climbs nine per step there, though the height is two. at zero, how much will the slope read? |

### dlg-00005-010 turn 4

**Pupil:** less than nine, since the curve is lower there.

**Tape tutor:** a natural guess, since the curve sits at minus two there, well under its height of two. if lower meant gentler, then I expect it to climb less than nine for each one across. well, the slope is back at nine, so at minus two the curve climbs as steeply as at two, only lower down. so height and steepness are separate things. at two and a half, how far past nine will it climb?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, less than nine. minus two cubed is minus eight, and minus six twos is two, so I hold the curve sits at minus two there. so I make its climb there about three and a half per step, less than the nine at two. the slope fell from nine to three and a half, so the curve steepens from two to minus two. at zero, where the height is zero, how much will the slope read? | right, less than nine. at minus two, I take away two from eight, so the curve stands at minus two. so I hold it climbs nine for every one across there, just as you said. the slope fell from nine to minus two, so the curve steepens as it drops. at zero, the very top, how much will the slope read? |
| A | right, less nine. minus two cubed is minus eight, and taking away six leaves two, so I hold the curve sits at two there. so I make the climb there nine point nine nine per step, less nine. the slope fell from nine to zero, so the curve lies flat at its lowest point. at zero, where the height is zero, how much will the slope read? | yes, less than nine, because the curve is falling there. at minus two, I take away two from eight, so the curve stands at minus two. so I hold it falls nine for every one across there. the slope fell from nine to minus two, so the curve steepens as it drops. at zero, where the curve crosses the middle, how much will the slope read? |
| B | right, less than nine. minus two cubed is minus eight, and taking away six leaves two, so I hold the curve sits at two there. so I make the climb there about two for each one across, a drop of eight in a step of one. the slope fell from nine to two, so the curve steepens from two to minus two. at zero, where the height is zero, how much will the slope read? | yes, less than nine, because the curve is lower there. at minus two, I take away two from eight, so the curve stands at minus two. so I hold that it falls nine for every one across there. the slope fell from nine to minus nine, so the curve steepens as it drops. at zero, where the curve crosses the middle, how much will the slope read? |

### dlg-00005-010 turn 6

**Pupil:** fifteen point seven five, since three times two point five squared, minus three.

**Tape tutor:** right, fifteen point seven five, and your working is sound. since two and a half sits at the edge, I make the curve there climb almost sixteen for every one across. the slope finishes at fifteen point seven five, so out at the edge the curve climbs nearly sixteen for each step of one. stepping back from two and a half to two point four, about how much will the height drop?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, fifteen point seven five, and that number comes from three times two point five squared, less three. so I hold that at two and a half the curve climbs fifteen point seven five for each one across. so the slope reads fifteen point seven five, past nine. at zero, where the curve crosses the middle, how much will the slope read? | right, minus three and a half, since three times two and a half is minus three and a half. so I hold that the curve at two and a half climbs about three and a half for each one across. the slope is back at three and a half, so the curve climbs three and a half for each one across at two and a half. at zero, how much will it climb for each |
| A | right, fifteen point seven five, and that number comes from three times two point five squared, less three. so I hold that at two and a half the curve climbs fifteen point seven five for each one across. so the slope is fifteen point seven five, more than nine, and that number changes from place to place. at zero, where the curve crosses the middle, how much will the slope read? | right, 15.75, since three times two and a half is 16.5, less three. so I hold that the curve at two and a half climbs 16.5 for each one across. the slope is back to nine, so the curve climbs nine for each one across at two and a half, just as steeply as at two. so the derivative is the slope at a place, not the height. what will the slope read |
| B | right, fifteen point seven five, since three times two point five squared is fifteen point nine six five, less three. so I hold the curve at two and a half climbs fifteen point seven five per step, more than nine. the slope reached fifteen point seven five, so the curve climbs harder at two and a half than at two. so the derivative is the slope at each place, and it changes from place to place. at zero, where the curve crosses the middle, how | right, 15.75, since three times two is six, less three. so I hold that the curve at two and a half climbs 15.75 for each one across. the slope is 15.75, so the curve climbs that hard there, though its height is only two and a half. at minus one and a half, how far past nine will it climb? |

### dlg-00005-010 turn 8

**Pupil:** zero point nine, since lower down it's gentler, like at two.

**Tape tutor:** near two a tenth cost about nine tenths, so it's a fair guess. if it holds here, I expect the curve to lose under one. it lost one and a half, so this step moves it harder, and that changing rate is the derivative. so where did a tenth move the height more, near two or at the edge?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, about nine tenths. I hold the curve at two point four climbs about nine tenths for each one across, since it sits lower than two and still climbs hard. the slope fell from fifteen point seven five to nine tenths, so the curve drops about one per step there. so the derivative is the slope, and it changes from place to place. would a straight | right, zero point nine, and your reasoning is sound. the curve at two point four sits lower than its hump, so I hold it must climb less steeply there. the slope fell from nine to zero point nine, so the curve steepens as it drops. so at two point four, how much will the curve climb per step across? |
| A | right, about zero point nine, since two point four sits lower than two and a half. so I hold the curve at two climbs nine for each one across, and at two point four about zero point nine for each one across. the slope fell from nine to zero point nine, so the curve is gentler there, though still rising. so the derivative is the slope at a place, and | right, zero point nine, and your reasoning is sound. if the curve at two point four fell as gently as at two, I'd hold it climbs less steeply there. the slope is zero point nine nine, so the curve at two point four climbs nearly as steeply as at two, though lower down. so at three, how much will the slope read? |
| B | right, about nine tenths, since two point four sits lower than two and a half. so I hold the curve at two point four climbs about nine tenths for each one across. the slope fell from fifteen point seven five to nine tenths, so the curve at two is gentler than at its edge. so the derivative is the slope at each place, and it changes from | right, 0.9, and your reasoning is sound. the curve at two point four sits lower than its hump, so I hold it climbs less steeply there. the slope fell from nine to 0.9, so the curve at two climbs nine for each one across, and at two point four about nine for each one across. so the steepness changes from |

### dlg-00005-027 turn 0

**Pupil:** a derivative's just the slope of a curve? one number.

**Tape tutor:** a slope, yes, but whether it's one number everywhere is something to test. it needs testing, because steepness belongs to a place, so we start with one. take x as zero, sine of zero is zero, and twice that is zero, so I expect this wave through the middle there. with zero height at that input, we have our wave sitting right on the axis. if x moves to one, how high will the wave stand?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | it's a fair question, so let's test | it's a fair guess, so let's |
| A | it's a natural thing to guess, since a curve is how fast it moves, so let's test it with numbers. I expect this cubic to climb to a hump, then fall to a dip, since it dips below zero in the middle. at input one, I get a cubic that sits above zero on the left and below on the right, so its steepness must shrink from one to nothing. | (silence) |
| B | it does, and numbers can settle that. we should find this curve rising and falling gently, since a straight line would keep rising and falling the same amount. at zero, the sine of zero is zero, so we should find the curve crossing the middle there. so how steep do you guess the curve is at zero? | (silence) |

### dlg-00005-027 turn 2

**Pupil:** one point six eight, since it's twice sine of one.

**Tape tutor:** yes, one point six eight is right, it is twice sine of one. I expect the wave stands that high above one, from your arithmetic. so we get a rise of one point six eight for a step of one. we find a tenth instead gives twice sine of a tenth, about point two, so there the rise is twice the step. so how steep, as a number, will the wave be right at x zero?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, twice sine of one is about one point six eight, so we should find the wave that high there. so the height rose from zero to one point six eight, a step of one across. I take it the wave climbs there, steep at one. the slope is one, so the wave climbs one for one across. at the peak, x two point five seven, how much will the slope read? | yes, one point six eight, and the sine of one is about one point six eight, so I hold the wave stands just under the axis there. the slope is zero, so the wave climbs one for every step along at one. at x zero, how steep do you guess the wave climbs? |
| A | right, twice one is two, so I hold the wave stands two high at one. so the height rose twice the step from zero to one, and I take it the wave is level at one. the slope is zero there, so the wave is flat at its top. if x moves to one point five, how much will the slope read? | yes, 0.68, and twice sine of one is about 0.96, so I put the wave at about one point nine six there. so we have the wave climbing from the axis to about one point nine six there. the slope is one, so the wave climbs one for each step along at x one. at x zero, how steep will the wave be? |
| B | right, twice sine of one is about two, so I hold the wave stands two high there. so the height rose twice the step from zero to one, and I take it the wave is steep there. the slope is two, so the wave climbs two per step there, while its height only went from zero to two. at x equals one point five, how much will the slope read? | yes, 0.68, and that number is the one we should find at one. twice sine of one is about 0.98, so I put the wave at about 0.98 there. so the height is nearly one, and I expect the wave to be almost as steep as it is high. the slope is near one, so the wave climbs nearly one per step along. at x equals two, how steep will the wave be? |

### dlg-00005-027 turn 4

**Pupil:** two, because a tenth step gives twice that rise.

**Tape tutor:** two is right, because a tenth along rose two tenths. so at zero we find the wave climbing two for every one along. the slope reads two, your number exactly. a whole step rose only one point six eight, so at one I expect a flatter wave. at one the slope is down to one point zero eight, roughly half what it was at zero. so how steep will the wave be at its peak, x one point five seven?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, two, because a tenth of a step gave twice the step. so I hold the wave at zero climbs two for every one across. the slope is two, so the wave climbs twice as fast as it moves there. at x one point five seven, how much will the slope read? | yes, two, because a tenth of a step gave twice the step. so I hold the wave at zero climbs two for every one across. the slope is two, so the wave climbs two for every one across at zero. at x one and a half, how much will the slope read? |
| A | yes, two, because twice sine of zero is twice nothing, so we have the wave falling there. so the wave drops two for each one across at zero, and at one it climbs two for each one across. so the slope is two there, while the height is zero. at x one point five seven, how steep will the wave be? | yes, two, because twice the step is twice the rise. so I hold that the wave at zero climbs two for every one along. the slope is two, so the wave climbs two for every one along at zero. at x one and a half, how much will the slope read? |
| B | yes, two, because a tenth of a step gave twice sine. so I hold the wave at zero climbs two for every one across. the slope is two, so the wave climbs twice as fast as it moves there. at x one point five seven, how much will the slope read? | yes, two, because twice the step is twice the step. so I hold that the wave at zero climbs two for every one along. the slope is two, so the wave climbs two for every one along at zero. at x one and a half, how much will the slope read? |

### dlg-00005-027 turn 6

**Pupil:** zero, since the top of a wave is flat for a moment.

**Tape tutor:** zero is right, since at its top the wave has stopped climbing. I think it keeps flattening right up to that peak, because its steepness already fell from two at zero to one point zero eight at one. zero at the peak, then, so this one wave's steepness is no single number, it changes with the place. at x two, past the peak, how much will it fall for each one along?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, zero, because the top is where the wave stops climbing and turns down. so I expect the wave at one point five seven to lie flat, its climb from zero done. the slope fell from one point zero eight to zero, so the top is flat while the peak is where the whole climb ended. how | yes, zero, because the wave has stopped climbing at its top. one point five seven is the highest point on the wave, so I expect it to lie level there. the slope fell to zero, so the wave is flat at its top, while at one it was climbing two for every one along. so how steep will the wave be at x two, the right edge? |
| A | yes, zero, because the wave stops climbing at its top. so I expect the wave to lie level at one point five seven, its top. the slope fell from one point zero eight to zero, so the wave is flat at its peak, though it rose twice the step before. so how steep will the wave be at x two, the right edge? | yes, zero, because the wave has stopped climbing at its peak. one point five seven is where the wave stands highest, and I expect it to lie level there. the slope fell from one point zero eight to zero, so the wave is level at its top. so how steep will the wave be at x two, the right edge? |
| B | yes, flat for a moment, because the top is where the wave stops climbing and starts falling. so I expect the wave at one point five seven to lie level, then to turn downhill. the slope fell from one point zero eight to minus one point six four, so the wave falls one for every one along there. so how steep will the wave be at x two, the right edge? | yes, flat for a moment, because the wave has just passed its peak at one point five seven. the slope fell from one point zero eight to zero, so the wave is level at its top. so the slope fell from two to zero, so the wave is level at its top, though it rose two for every one along at its peak. how steep will the wave be at x two, the right edge? |

### dlg-00005-027 turn 8

**Pupil:** minus zero point eight three, because past the peak it's falling.

**Tape tutor:** yes, past its peak the wave comes down, so minus zero point eight three. we have it dropping nearly one for each one along at two. so the slope depends on the place, which is the derivative. what is a straight line's derivative?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, minus zero point eight three, because past the peak the wave is falling. I take it to be falling three times as hard as it climbed at one. the slope ran from one point zero eight at one, down to minus zero point eight three at two. so the derivative is the slope, and it changes | yes, minus point eight three, because past the peak the wave is falling. so I expect it to drop about one point eight three for each one along at x two. the slope is minus one point eight three, so the wave falls harder there than at its top. at x three, how much will |
| A | yes, minus zero point eight three, because past the peak the wave still falls for each step along. so I expect it to drop three tenths for each one along at x two. the slope is minus three point one three, so the wave falls harder there than at its peak. so the derivative is | yes, minus point eight three, because past the peak the wave falls harder than it rose. so I expect it to drop three for each one along at x two. the slope is minus three, so the wave falls three for each one along there, just as you said. at x three, how much will it fall for each one |
| B | yes, minus zero point eight three, because past the peak the wave still falls for each step along. so I expect it to drop about three tenths for each tenth along at x two. the slope is minus zero point eight three, so the wave falls there just as your rule said. so the derivative is that | yes, minus point eight three, because past the peak the wave is falling again. so I expect it to drop about three for each one along at x two. the slope fell from zero at the peak to minus three, so the wave falls three for each one along there. at x three, how much will it fall |

### dlg-00005-038 turn 0

**Pupil:** hello, so what does a derivative actually tell you about a curve?

**Tape tutor:** good question, so I'll find it with numbers, not a definition. height alone can't settle it, because a low curve can still climb fast. my bet is this cubic climbs, sinks, then climbs again, since the cube wins at both ends. at one, cubing gives one, and three comes off, so I make the height minus two, in its dip. if the dot steps right from one, how many times the step will the height move?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | it tells you how fast the height changes, so let's work one out. I expect this curve to climb to a hump, then fall to a dip. at zero, the sine of zero is zero, and twice zero is still zero, so I hold the curve sits at height zero there. if the input steps from zero to one tenth, how far will the height move? | it tells you how fast the height changes, so let's test that with numbers. I expect this curve to climb to a hump and then fall to a dip. at one, one times one is one, and one minus one is zero, so I hold the curve sits on the axis there. if the input moves to one point one, how far will the height move? |
| A | it tells you how fast the height changes, so let's test that with numbers. I expect this curve to climb to a hump, then sink to a dip. at zero, zero cubed is zero, and three times zero is zero, so I hold the curve sits at height zero there. if the input steps from zero to one, how much will the height change? | (silence) |
| B | it tells you how fast the height changes, so let's work one out. I expect this cubic to climb to a hump, then sink to a dip. at minus two, I get minus two times minus two, plus three, so I hold the curve at minus two to be at height minus two. if I nudge the input to minus one, how far will | (silence) |

### dlg-00005-038 turn 2

**Pupil:** up, but barely, since the dot sits in the bottom of the dip.

**Tape tutor:** yes, up and barely, since the dip is nearly flat. one point one cubed, less three point three, gives minus one point nine six nine, so I put the curve just a hair above its floor there. three hundredths for a tenth of a step means that at one itself, I reckon the curve has no tilt. the slope is zero, so the floor of the dip is level. at two, how many times the step will the slope read?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, barely, and the board can show it. if the dot moved right from one to one point one, I make the height just under minus two point nine nine. that step barely lifted the curve, so I hold the curve at one barely climbs, though it sits low. the slope is zero, so the curve climbs only because it turns out of the dip. at two, how many times the step will the height move? | right, and the height is the one number we can test on the step. if the dot sits at one point one, I make its height minus point one nine five, a drop of about a tenth for a tenth along. so I hold the curve at one falls about once the step, though its height is zero. the slope is zero, though the height is zero. at two, how much will the slope read? |
| A | right, and the step is tiny, so I hold the curve barely moves there. one point one cubed is one point three three one, less three, so I make the height minus two point nine seven one. that drop is about three tenths for a step of one tenth, so I take the curve to climb about three times its step at one. the slope is three, three up for each one along. at zero, where the curve crosses the middle, how many times the step will it move? | yes, it sits there, so I hold the curve must lie level at one. a tenth right of one cubes to a thousandth, so I make the height there near one point nine nine nine. so the curve lost about a hundredth of height between one and one point one, a tenth of step for a tenth of drop. at zero, how many times the step will the height move? |
| B | right, barely, since the dip is only a little below zero. I expect the curve to rise just under one per step at one, since the cube outgrows the three taken off. the slope is zero at the dip, yet the curve still climbs there. at two, how many times the step will the height move? | right, it sits there, so I hold the curve must lie level at one. the step is a tenth, so I claim the curve climbs about one per step there. the slope is one, so the curve climbs one for each step along. at zero, where the height is zero, how many times the step will the height move? |

### dlg-00005-038 turn 4

**Pupil:** two, because the curve sits at a height of two there.

**Tape tutor:** two is your call, and the height there is two, so I'll work it. two cubed is eight, and six off that leaves two, so I put the curve two high there. on your rule, we get the curve climbing two for each step out at two. instead it climbs nine for every step there, though it stands only two high. at zero, how many times the step will the slope read?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, two, because two cubed less three is minus one and a half, so I make the curve two high there. and since two point one cubed less three point two five is minus point two nine six nine, I put the curve climbing there just as hard as at one, though its height is two. the slope ran from zero to nine, so the curve climbs nine for each one across at two, where its height is two. at zero, how many times the step will the slope | yes, two, because two is the second height we have found. two cubed, less six, leaves minus two, so I put the curve below the floor there. two is the second height, so I take it the curve climbs two for every one along at two. the slope is two, so the curve climbs two for every one along at two. at three, how many times the step will the slope read? |
| A | yes, two, because two cubed less three is minus one and a half, so I make the curve sit at two there. two point one cubed, less six, leaves minus one point three nine nine, so I hold the curve at two sits lower than its floor. so the slope is two, two times the step, though the height there is minus one and a half. at zero, how many times the step will the slope read? | yes, two, since the curve has stopped sinking by two. two cubed, less six, leaves minus two, so I put the curve at minus two there. two tenths |
| B | yes, two, because two cubed is eight, less six, so I make the curve stand two high there. and since the floor was level, I hold that the curve steepens from level to two. the slope went from zero to eight, so the curve climbs eight times the step there, while at one it barely moved. at three, how many times the step will the slope read? | yes, two, since the curve has stopped sinking at one. two cubed, less six, gives minus two, so I put the curve at minus two there. |

### dlg-00005-038 turn 6

**Pupil:** minus three, since the curve sinks there, heading into the dip.

**Tape tutor:** yes, minus three. zero cubed less zero is zero, so I place the curve at height zero there. a tenth along gives minus point two nine nine, so I find it falling about three times the step. going back past the dip, I see its tilt drop through level to minus three. the slope ran from nine, through zero at the dip, to minus three. at minus one and a half, what will the slope read?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, minus three, because the curve falls there toward the dip. zero cubed is zero, and six off that leaves minus two, so I put the curve below the floor there. so I expect it to drop three for each step out at zero, as steep as at two, yet falling. the slope ran from minus three through zero to nine, so the derivative is the curve's steepness at each place, not its height. would a straight line give one slope everywhere? | yes, minus three, since the curve sinks there. zero cubed is zero, and six off that leaves minus six, so I put the curve below the floor there. the slope is minus three, so the curve sinks three for each step out at zero. at one, how many times the step will the slope read? |
| A | yes, three, because the curve heads down into its dip. zero cubed is zero, and six off leaves minus two, so I put the curve at ground level there. if your rule held, I'd expect it to sink three | yes, minus three, since the curve sinks there too. zero cubed is zero, and six off that is minus six, so I put the curve below the floor there. the slope is minus three, so the curve sinks three for each step out at zero. at one, how many times the step will the slope read? |
| B | yes, minus three, because the curve sinks there before its dip. zero cubed is zero, and six off leaves minus two, so I put the curve below the floor there. so I take it to sink there faster than it climbs at two. the slope fell from nine to minus three, so the curve falls faster there than it climbs at two. at three point four six, how many times the step will the slope read? | yes, minus three, because the curve sinks there too. zero cubed is zero, and six off that is minus six, so I put the curve below the floor there. so I |

### dlg-00005-038 turn 8

**Pupil:** one point one, since the curve stands that high there.

**Tape tutor:** a fair guess, and I'll try it, since height and steepness tangle easily. we hold the curve at minus one and a half climbs far harder than its height. it climbs three point seven five per step, so steepness never followed height. which was bigger there, the height or the slope?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | one point one is right, and the height there is one point three nine five, so I put the curve just under the floor of the dip. if your rule held, I'd expect the curve to climb about one | yes, one point one, since the curve stands that high there. one point one cubed less three point one five gives minus one point nine nine, so I put the curve that high. the slope is one point one, so the curve climbs one for each step along there. the slope ran from minus three to one point one, so the derivative |
| A | one point one is right, and the height is the reason. minus one and a half cubed is minus one and a quarter, so I put the curve above the floor there. if your rule held, I'd expect it to climb there | yes, 0.1, since the curve stands that high there. one and a half cubed less six is minus one and a quarter, so I put the curve that high. the slope is zero there, so the curve climbs three for each step along at minus one and a half. the slope ran from zero at the dip to zero at |
| B | yes, minus one and a half, since the curve stands there at minus one and a half. a tenth along gives minus point two nine nine, so I put the curve falling about as hard as at zero, yet higher. the slope ran from minus three to minus one and a half, so the derivative is the curve's steepness at each place, not | yes, one point one, since the curve stands that high there. one point one cubed less three point one five gives minus one point nine nine, so I put the curve that high there. the slope fell from minus three to zero, so the curve climbs there while still standing high. at minus two, how many times the step will the slope read? |

### dlg-00005-056 turn 0

**Pupil:** is a derivative just how high a graph is, or not quite?

**Tape tutor:** not quite, and one input pushed through our bowl can show why, because numbers settle it better than opinions. squaring never turns negative, so we know this bowl bottoms out right at its middle. one times one is one, less one leaves zero, so we must find the bowl at height zero above x equals one. at the dot on x equals one, where the height is zero, will the slope there also come out zero, or another number?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | it's how fast a graph moves, so let's test that on numbers first. I expect this cubic to climb to a hump, then fall to a dip, since it never climbs as steeply as it falls. | it's a fair guess, since a derivative is about how fast a curve changes, so we should find one curve that climbs and falls at the same rate. I expect this curve to climb to a hump and then fall back to zero, since a hump is the highest point on any curve. at zero, one times zero is zero, and one minus zero is one, so I hold the curve sits one high there. if the input steps from zero to one, how far will the height drop? |
| A | it's a natural guess, since a hump stands out on a wave. so we should find this wave highest at its middle, since its rise there is gentlest. at the input two, we get two times the sine of two, about two and a half. so we should find the wave two and a half high at two, since its rise there is gentlest. the output is two and a half, so the wave is level at its middle, as we guessed. at | (silence) |
| B | it's both, because height is the number on the board, and steepness is how fast it changes. I expect this cubic to climb fastest where its hump stands highest, since the hump lifts the whole curve out of the valley. | (silence) |

### dlg-00005-056 turn 2

**Pupil:** two, since the bowl is still climbing there.

**Tape tutor:** yes, two, and one more input shows where it comes from. one point one squared is one point two one, less one, so I reckon the bowl sits nought point two one high there. a tenth across bought two tenths up, so I'm betting the bowl climbs two per step at one. so the slope is two, where the height is zero. how much will the slope read at the lowest point, x equals zero?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, two, because the bowl still climbs at one, though its height there is zero. so we have the bowl climbing two for every one along at x equals one, though its height there is zero. the slope is two, while the height is zero, so the two are different numbers. at x equals two, will the slope read two again, or something else? | yes, two, because the bowl is still climbing there. one point one times one point one is one point two one, less one leaves zero point two one, so we find the bowl two hundredths high above x equals one. so the slope is two, twice the step we took. at x equals zero, where the height is zero, how much will the slope read? |
| A | right, two, because the bowl is still climbing at one. so I hold that the bowl must be rising two for every one along at one, though its height there is zero. the slope is two, while the height is zero, so the two are different numbers. at x equals two, will the slope read two again, or something else? | yes, two, because the bowl is still climbing there. one point one times one point one is one point two one, less one leaves zero point two one, so we find the bowl two hundredths high above x equals one. so the slope is two, twice the step we took. at x equals zero, where the height is zero, how much will the slope read? |
| B | right, still climbing, so we can't find the bowl at x equals one lying flat. one point one squared is one point two one, less one leaves zero point two one, so I claim the bowl stands two point one high above x equals one. the slope is two, so the bowl climbs two per step there, though its height is zero. at x equals two, will the slope read two again, or something else? | yes, zero, because the bowl is still climbing from its lowest point there. one point one squared is one point two one, less one is zero point one one, so we find the bowl one point one one high above x equals one. so the slope is one, one up for one along, just as you said. at x equals zero, where the height is zero, how much will the slope read? |

### dlg-00005-056 turn 4

**Pupil:** minus one, since the bowl sits one below zero there.

**Tape tutor:** minus one is the height there, and the numbers can test your call. zero gives minus one, a tenth gives minus nought point nine nine, so the bowl, as I work it, lifts just a hundredth. a hundredth per tenth is almost nothing, which leaves us a bottom lying flat. the slope came to zero there, while the height stays minus one. at x equals a half, how much will the slope read?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, minus one, because the bowl is one below zero there. zero times zero is zero, less one leaves minus one, so we must find the bowl at its lowest point there. and since the bowl stops climbing at one, I'd say it must be falling here. the slope fell from two to minus one, so the bowl drops one per step at zero. how much will the slope read at x equals two, the right edge? | yes, minus one, because the bowl is lowest at its middle. zero squared is zero, less one is minus one, so I put the bowl at minus one there. a tenth across bought about a tenth up, so I reckon the bowl drops one per step at its lowest. so the slope fell from two to minus one, where the height is zero. how much will the slope read at x equals one, the right edge? |
| A | yes, one, because the bowl stops falling just short of its lowest point. zero squared is zero, less one leaves minus one, so I put the bowl's lowest point at minus one and a half. a tenth on, at x equals zero point one, I'd claim the bowl climbs back to zero. so the slope fell from two to zero, while the height fell from minus one and a half to zero. how much will the slope read at x equals two, the right | yes, one, and the height of minus one shows why. one less one is zero, so we have the bowl at ground level there. a tenth across bought about a tenth up, so I reckon the bowl climbs one per step at zero. so the slope fell from two to zero, where the height is zero. how much will the slope read at x equals one, the right edge? |
| B | yes, minus one, because the bowl dips below zero at its bottom. at x equals zero, one times one is one, less one is zero, so we find the bowl's lowest point on the right edge. so I take it the bowl falls one per step there, while still lying flat. the slope fell from two to minus one, so the steepness changed from place to place. how much will the slope read at x equals two, the right edge? | yes, minus one, because the bowl is lowest at its middle. zero point one squared is one point two one, less one, so I put the bowl at minus one point two one there. so we have it falling one per step at its lowest point. the slope fell from two to minus one, so the bowl steepens as it gets lower. how much will the slope read at x equals two, the right edge? |

### dlg-00005-056 turn 6

**Pupil:** one, since the slope doubles x and x is zero point five.

**Tape tutor:** yes, one, your doubling rule holds. a half squared is a quarter, less one, so I put the bowl at minus nought point seven five, and nought point six gives minus nought point six four. eleven hundredths per tenth, so I take the bowl to climb about one per step. the slope reads one there, while the height stays minus three quarters. how much will the slope read at x equals one and a half?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, one, because the bowl doubles x at a half. a half squared is a quarter, less one, so I put the bowl at minus three quarters there. a quarter per step is more than the hundredth we got at zero, so I reckon the bowl climbs faster there. the slope went from zero to one, so the bowl climbs one per step at a half. how much will the slope read at x equals two? | yes, one, and the numbers can settle it. a half squared is a quarter, less one is minus three quarters, so I put the bowl at minus three quarters there. a tenth across gave three tenths, so I reckon the bowl climbs three per step at a half. the slope went from zero to three, so the bowl climbs three per step at a half. how much will the slope read at x equals one?+ |
| A | yes, two, because the bowl doubles its steepness as it goes out. a half squared is a quarter, less one is three quarters, so I put the bowl at three quarters high there. a tenth across gave three tenths up, so I reckon it climbs four per step at a half. the slope went from zero to four, so the bowl doubles its steepness as it goes out. how much will the slope read at x equals two? | yes, and the numbers can test that. a half squared is a quarter, less one is a third, so I put the bowl at height three there. a tenth across gave three tenths, so I reckon it climbs three per step there. the slope is three, where the height is three, so the derivative is the steepness at each place. at x equals one, how much will the slope read? |
| B | yes, one, because a half squared is a quarter, less one is a third, so I put the bowl at a third of its height there. a tenth across gave a tenth of a third, so I reckon the bowl climbs one per step at a half. the slope went from zero to one, so the bowl climbs one per step at a half, where its height is a third. at x equals two, how much will the slope read? | yes, zero point five, and the numbers show why. a half squared is a quarter, less one is a third, so I put the bowl at height three there. a tenth across gave three tenths, so I make the bowl climb three per step at a half. the slope went from zero to three, so the bowl climbs three per step at a half, where its height is three. at x equals one, how much will the slope read? |

### dlg-00005-056 turn 8

**Pupil:** one point two five, since the bowl sits that high there.

**Tape tutor:** one and a half squared, less one, and I get the bowl one and a quarter high there. if steepness were height, I'm guessing the bowl tilts one and a quarter there. three, so steepness and height part ways, as everywhere we worked. what did the slope read at one and a half?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, one and a quarter, your height is right. one and a tenth squared is one and twelve hundredths, less one, so I put the bowl at one and a quarter there. a hundredth over a tenth is one per step, so I take it to climb one and a quarter per step there. the slope reads one and a quarter, while the height is one and a quarter. | yes, one point two five, and the numbers show why. one and a half squared is two and a quarter, less one, so I put the bowl at minus one and a quarter. a tenth across gave minus one point nine nine, so I take it to drop about one per step there. the slope is one point two five, so the bowl drops one per step there, though its height is minus one and |
| A | yes, one and a half squared is two and a quarter, less one, so I put the bowl at minus nought point two five. a tenth gave two hundredths, so I take it to climb two per step there. the slope reads two and a quarter, while the height is minus nought point two five. at x equals two, how much will the slope read? | yes, 0.25, and the numbers show why. one and a half squared is two and a quarter, less one is minus one and a quarter, so I put the bowl at minus one and a quarter there. a tenth across gave minus one and a tenth, so I take it the bowl drops one per step there. the slope is minus one, the height is minus one and a quarter, |
| B | yes, one point two five, and the bowl itself shows why. one and a half squared is two and a quarter, less one, so I put the bowl at minus one and a quarter. a tenth on, it's minus one point two nine, so I reckon it climbs one point two five per step there. the slope reads one point two five, where the bowl sits at one and a half. | yes, one point two five, and the numbers can show why. one and a half squared is two and a quarter, less one is minus one and a quarter, so I put the bowl at minus one and a quarter there. the slope is one and a quarter, so the bowl climbs one and a quarter per step there, while it sits at minus one and a quarter. the slope went from zero at |

### dlg-00005-063 turn 0

**Pupil:** hi there, a derivative is about steepness. steeper means higher, right?

**Tape tutor:** it's a natural hunch, since steep climbs end high, but height and steepness are separate numbers. I'd expect, by your rule, the wave to be gentlest where its height is zero, midway between top and bottom. the sine of zero is zero, and twice zero is zero, so we hold the height is zero where x is zero. if we step x to zero point one, how much will the height rise?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | let's test that with numbers. I expect a bowl that climbs at one and falls at minus one, since cubing outgrows doubling fast. at input two, I get a height of four, and two down from that leaves one. so I hold that the bowl sits at height one above input two. if the input moves to two point one, how far will the height drop? | yes, steepness is about height, so let's test that on numbers. we should find this curve steepest where its hump stands highest. at zero, the cube is zero, and one minus three zeros is three, so we have the curve three high there. at one, the cube is one, and one minus three ones is minus two, so we have the curve two high there. so the height fell one for |
| A | yes, steepness is about height, so let's test that on numbers before any rule. I expect this curve to climb, dip, then climb again, because its cube and three times the input both keep growing. at input zero, the cube is zero and three times zero is zero, so I hold the curve sits at height zero there. if the input moves to one tenth, how far will the height move? | (silence) |
| B | yes, steeper does mean higher, so we should find this curve climbing fastest where it lies highest. at zero, the sine is zero, so we have the curve sitting at height zero. if the input moves to one tenth, how much will the height change? | yes, steeper does mean higher, so we should find this curve climbing fastest where it lies lowest. at minus two, I expect the curve to rise two for every one along, since two times two is four. so I hold that the curve lies lowest at minus two, where it climbs two for every one along. the slope is two, so the curve climbs two for every one along at minus two. at zero, where the curve crosses |

### dlg-00005-063 turn 2

**Pupil:** about zero point two, since sine of small x is nearly x.

**Tape tutor:** yes, because the sine of a small x is nearly x itself. so we have the curve standing zero point one nine nine seven high at zero point one. since the height moved about twice the step, I'd bet the curve at zero climbs two for each one across. the slope is two, the very ratio our step gave. at the peak, x one point five seven, how many times the step will the height move?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, about two tenths, since sine of zero point one is nearly one tenth. so I hold the wave stands two tenths higher at zero point one than at zero. the height went from zero to two tenths, so the wave climbed two for every one across. at x one, how much will the height rise? | yes, about zero point two, since the sine of zero point one is nearly zero point two. so we hold the wave stands nearly two tenths high at zero point one. the slope is two, so the wave climbs two for each step along at zero. at x zero, how much will the slope read? |
| A | yes, about zero point two, since the sine of zero point one is nearly one tenth. so I hold the wave stands two tenths high at zero point one, double its sine. the slope is two, so the wave climbs two for each step across there. at x zero, where the height is zero, how much will the slope read? | yes, about zero point two, since the sine of zero point one is nearly zero. so we hold the wave stands nearly as high as it is wide at x zero point one. the slope is zero, so the wave is level at its top. at x one, how much will the slope read? |
| B | yes, about zero point two, since the sine of zero point one is nearly zero point one. so we should find the wave barely higher at zero point one than at zero, since its climb is barely started. the height went from zero to zero point two, so the wave rose twice the step we took. at x equals one, how much will the height rise? | yes, about zero point two, since the sine of zero point one is about zero point two. so we hold the wave stands two tenths high at x zero point one. the slope is two, so the wave climbs two for each step along at that place. at x zero, where the height is zero, how much will the slope read? |

### dlg-00005-063 turn 4

**Pupil:** two times, since the wave crosses zero there going down.

**Tape tutor:** you say two, so I'll work the peak like zero. twice the sine of one point five seven is two, and one point six seven gives one point nine nine, so I find the height barely moves. that's a tenth of the step, so I take the curve as nearly level on top. the slope fell from two to zero. at x one, partway up, how many times the step will the height move?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, two, because the wave crosses zero there on its way down. so I hold the wave climbs one and a half times its step at x one point five seven, against zero at x zero. the slope fell from two to zero, so steepness changes from place to place. at x two point five, how many times the step will the height move? | yes, two, because the wave crosses zero height at one point five seven. the slope fell from two to zero, so the wave steepens as it moves toward its top. the slope is zero at the top, so the wave is flat there, yet it climbs two for each one across at zero. at x two, how many times the step will the height move? |
| A | yes, two, because the wave crosses zero at its top. twice the sine of one point five seven is two, and that gives the top a height of two. so I hold the wave falls two for each one across there, as steep as it climbed two. the slope fell from two to zero, so the steepness changes from place to place. at x two, how many times the step will the height move? | yes, two, because the wave crosses zero height at the peak. the sine of one point five seven is one point six nine, and twice that is three point three nine eight. so we have the wave at its steepest there, though its height is zero. the slope fell from two to zero, so the steepness is where the height is. at x two, how many times the step will the height move? |
| B | yes, two, because the wave crosses zero there on its way down. so I hold the wave climbs two for each one across at the top, just as steep as at zero. the slope fell from two to zero, so the top is flat, yet it climbs just as hard as zero. at x two point five, how many times the step will the height move? | yes, two, because the wave crosses zero at its top. the height there is zero, so we have the wave falling two for each one across. the slope fell from two to zero, so the steepness changed from place to place. at x two, how many times the step will the height move? |

### dlg-00005-063 turn 6

**Pupil:** less than two times, since the slope shrinks toward the peak.

**Tape tutor:** right, less, since the climb eases near the top. twice the sine of one is one point six eight, so we put the curve that high at x one. having eased, I'd guess the curve at one climbs about half as steeply as at zero. the slope reads one point zero eight, so a tenth across lifts about a tenth. past the top, at x two point five, how far will a tenth across shift the height?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, less than two, because the climb shrinks toward the top. I hold the curve at one climbs under two per step, since it has already lost its flat top. the slope fell to zero point nine nine, so the curve climbs under two per step there. at x two, how many times the step will the height move? | yes, less than two, because the curve is level at the top. one point six seven squared is two point two nine, and one point six seven takes away two point two nine, so we have the curve at minus one point nine nine. so I hold it climbs one for each step across at one. the slope fell from two to zero, so the derivative is the slope at each place, and that number changes from place to place. what rule gives the slope at any x? |
| A | yes, less than two, because the slope shrank as the peak came near. I hold the curve at one climbs under one per step, since its top is flat. the slope fell to zero, so the curve climbs under one per step there. at x two, how many times the step will the height move? | yes, less than two, because the curve is level at one. twice the sine of one is one, and one point six seven is two, so I hold the curve at one sits two high. the slope fell from two to zero, so the curve is level at one and climbing two for each one across. the slope is zero at the top, so the derivative is the slope at each place. what will the slope read at x two? |
| B | yes, less than two, because the slope shrank as the peak came into view. I hold the curve at one climbs under one per step, since it has already lost its flat top. the slope fell to zero, so the curve at one climbs nothing per step. at x two, how many times the step will the height move? | yes, less than two, because the curve is level at one. twice the sine of one is one, and one point six seven is two, so I put the curve at height two there. the slope fell from two to zero, so the curve is level at one and steep at zero. at x two, how many times the step will the height move? |

### dlg-00005-063 turn 8

**Pupil:** down, since past the peak the curve falls.

**Tape tutor:** right, it falls, so the height goes down. I'll stake it that the curve at two point five loses more than one for each one across. the slope is minus one point six, though the height still stands at one point two. so each x gave its own number, two, zero, minus one point six, and that number for each x is the derivative. could a curve's derivative stay one number everywhere?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, down, since past the peak the curve turns back toward zero. twice the sine of two point five is one point six eight, so I hold the curve sits below zero there. a tenth across dropped the height two tenths, so I take the curve at two point five to fall about twice as hard as it climbed at zero. the slope ran from zero to minus two, so steepness changes from place to place, and that changing number is the derivative. | yes, down, because past the top the curve falls harder than it climbed. twice the sine of two point five is minus one point nine nine, so we have the curve falling one point nine nine per step there. so I take it to be falling about one per step at x two point five. the slope is minus one, so the derivative is the slope, and that number changes from place to place. |
| A | yes, past the top the curve falls, and a tenth across at two point five is a fair test. twice the sine of two point five is one point six eight, so I make the height there minus one point nine nine. so I hold the curve drops about twice the step there. the slope fell to minus two, so the derivative is steepness, and that number changes from place to place. would a straight line keep one slope everywhere? | yes, down it goes, since the curve has passed the top. two point five seven times two is five, so I hold the curve at minus two point five seven there. so I take it to fall about one per step past the top. the slope is minus one, so the derivative is the slope, and that number changes from place to place. what does the slope read at x zero? |
| B | yes, down, because past the top the wave has turned away from its peak. twice the sine of two point five is one point six nine, so we put the wave below zero there. so I'd say it drops about twice the step there, as at zero. the slope fell to minus two, so the derivative is the slope, and it changes from place to place. would a straight line keep changing like that? | yes, down, because past the top the curve has already started to fall. twice the sine of two point five is one point nine nine, so I hold the curve below zero there. a tenth across drops it about a tenth, so I take the curve to fall about one per step past the top. the slope is minus one, so the derivative is the rate of change, and that number changes from place to place. what rule gives the slope at |

### dlg-00005-073 turn 0

**Pupil:** a derivative is just the slope? a curve has one slope everywhere?

**Tape tutor:** a slope, yes, but whether it's one slope, numbers can test. one height shows no steepness, so we compare two close ones. doubling just stretches sine, so we know this wave swings evenly. at zero, sine is zero, and doubled stays zero, so our wave crosses the middle there. a tenth along, sine is nearly a tenth, so I put the wave two tenths up. at one point five seven, will a tenth step lift the height two tenths again?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | let's test that with numbers on one wave. I expect this wave to climb and fall twice for every step along, because its sine doubles twice. at input zero, sine of zero is zero, and doubled it's still zero, so I hold the wave crosses the middle there. if the input moves from zero to a tenth, how far will the output move? | it does, and numbers can test that. a cube has one peak, because squaring anything takes away one thing from it. we should find this curve climbing to a hump and then falling back to zero. at one, one times one is one, and one minus one is zero, so we have the curve sitting on the axis there. if the input moves to one point one, how far will the height move? |
| A | right, so I expect this curve to climb and fall in equal steps. I hold that at zero, the sine of zero is zero, and twice zero is still zero. so I say the curve crosses the middle at zero, at height zero. if the input moves to one tenth, how much will the output move? | (silence) |
| B | it does, and that slope is what tells us how fast the height changes. I expect this cubic to climb to a hump, then fall to a dip. at minus two, I get minus two times minus two, plus three, so I hold the curve sits at minus two there. if the input moves to minus one, how far will the height move? | (silence) |

### dlg-00005-073 turn 2

**Pupil:** yes, two tenths again, because the curve keeps one slope.

**Tape tutor:** two tenths held near zero, so your guess earns its trial up high. at one point five seven, we have the wave at its peak, height two. on one slope, we need the wave a tenth further on to sit two tenths above that top. the second height reads one point nine nine, so up here a tenth step dropped the wave a hundredth. how steep would you guess the wave is at one point five seven?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, two tenths, because the curve keeps one slope. so I hold that at one point five seven the wave climbs two for every one across. the slope is two, so the wave climbs twice as fast as it moves. at the peak, one point five six, how much will the slope read? | yes, two tenths, because the wave is still level at one point five seven. the slope is zero there, so the wave climbs one for every step along. the slope ran from zero to one, so the wave climbs one per step there. at two, how much will the slope read? |
| A | yes, two tenths, because the wave keeps one slope. so I hold that at one point five seven the wave stands two tenths up, just as the tenth before did. the slope is two, so the wave climbs two for each step along. at the peak, x one point five seven, how much will the slope read? | yes, two tenths, since the wave is still level at one point five seven. the slope is zero at the top, so the wave is level there, though it stands two tenths high. the slope is zero, so the wave is level at its highest point. at minus one point five seven, how much will the slope read? |
| B | right, two tenths, because the curve keeps one slope. so I hold that at one point five seven the wave lifts two tenths for each tenth along. the slope is two, so the wave climbs two for each one along there. at the peak, one point five six, will the slope read two again? | yes, two tenths, because the wave is still level at one point five seven. the slope is zero at the top, so the wave is level there, though it climbs two tenths over a tenth. the slope is zero, so the wave is level at its highest. at minus one point five seven, how much will the slope read? |

### dlg-00005-073 turn 4

**Pupil:** zero, because at the peak the wave is flat.

**Tape tutor:** zero holds, since a tenth past the top the wave dropped only a hundredth. so at one point five seven, we have a wave whose steepness comes to nothing. yet near zero the wave rose two tenths for every tenth, so there we have a wave twice as steep as it runs. the slope climbed from nothing to two, so one wave carries many slopes. what will the slope read at three point one four?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | a flat top is a natural guess, so let's test it. if the wave were flat at its peak, I'd expect a tenth step barely to move it. the second height is one point nine nine, so the wave dropped a hundredth in a tenth. I take it the wave at one point five seven falls one for every one across. the slope is minus one, so the wave falls one for every one across at its top. at the peak, how much will the slope read? | zero holds, so we test it with numbers. at one point five seven, the wave is flat, so we expect no drop per step there. the slope is zero, so the wave is level at its top, though it stands two high. at minus one, how much will the slope read? |
| A | flat is a fair guess, since the top is level. a tenth further on, at one point six seven, I get the wave near minus one point nine nine. so I take it the wave falls here as steeply as it rose at the peak. the slope is minus one point three nine, so the wave falls here as steeply as it rose. at the peak, height two, I get a slope of zero. at the crossing, height zero, how much will the slope read? | yes, zero, because the wave has stopped rising by one point five seven, so we have it level at one point five seven. |
| B | flat is right, so we can test that with numbers. a tenth past the peak, sine is near one, so I put the wave near its top there. a tenth further, it has dropped to about zero, so I take the wave to be falling one for one past its top. the slope reads minus one, so the wave falls one for every one along there. at the peak, height two, did the slope read zero, or minus one? | zero holds, so we test it with numbers. at one point five seven, the wave is flat, so I put its steepness at zero. the slope is zero, so the wave is level at its top. at zero, the wave is zero high, so I put its steepness at minus zero. how steep do you guess it is at minus one? |

### dlg-00005-073 turn 6

**Pupil:** minus two, since the wave falls steeply through the middle there.

**Tape tutor:** minus two holds, and arithmetic earns it. doubled sine a tenth past three point one four is minus two tenths, so I place the wave just under the middle there. the second height now reads minus point two, two tenths down. so I'm betting it falls there as steeply as it rose at zero. the slope lands on minus two, two down for each one. at one, will the slope read nearer two or nearer one?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | minus two holds, and your reason fits the wave we have. at three point one four, sine is zero, and doubled it stays zero, so the wave crosses the middle there. a tenth along, it's near minus two tenths, so I put the wave falling two for each one across. the slope went from zero to minus two, so the wave falls twice as fast as it runs. how steep will the slope read at minus one point five? | minus two holds, since the wave falls steeply there. at three point one four, I put the wave falling two for every step along. the slope reads minus two, so the wave drops two for every step along there. so the slope changes from place to place, and that changing number is the derivative. what will the slope read at minus one? |
| A | two holds, since the wave falls steeply through its middle. at three point one four, sine is near one point nine seven, so doubled it is two. so I put the wave below the axis there, falling two for every one across. the slope reads minus two, so the wave drops two for each one across there. how much will the slope read at zero? | two holds, since the wave falls steeply in the middle. at three point one four, I put the wave falling two for every step along. the slope reads minus two, so the wave drops two for every step along there. so one wave has many slopes, and those slopes change from place to place. what will the slope read at minus one? |
| B | minus two holds, since the wave falls twice as hard as it rises at zero. at three point one four, I make the wave sit below the axis, two tenths down for every tenth along. so I take it to be falling there at two per step. the slope came to minus two, so the wave drops two for each step along. how much will the slope read at minus one point five? | minus two holds, since the wave falls steeply there. at three point one four, we have the wave falling two for every step along. the slope reads minus two, so the wave falls two for every step along there. so the derivative is the slope at a place, and that place is three point one four. how steep will the wave read at minus one? |

### dlg-00005-073 turn 8

**Pupil:** nearer one, since one sits closer to the flat peak than zero.

**Tape tutor:** nearer one is right, since one sits close to the flat peak. so I'll claim the wave at one climbs about one for each step along. there the wave rises one point zero eight for each step, as you guessed. so the derivative is a rule giving each place its own slope. would three sine x keep these same slopes?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | nearer one holds, and the numbers show why. at one, doubled sine is about one point six eight, so I put the wave just under its top. so I claim it falls there about as gently as it climbs at zero. the slope eased from minus two to about zero, so the wave falls gently at one and climbs steeply at zero. how much will the slope read at minus one point | yes, nearer one, since the wave is level at one. the slope fell from minus two to zero, so the wave lies level at one. so the slope fell from two to zero, so the wave is level at one and falling at one. the slope reads zero at one, so the wave is level there. at minus one point five seven, will the slope read near zero or near minus two? |
| A | nearer one holds, and the numbers show why. at one, doubled sine of one is zero, so I put the wave on the middle line there. a tenth on, it's twice sine of one tenth, so I claim the wave climbs there. the slope went from minus two to about one point nine nine, so near one the wave climbs near one per step. so the slope changes from place to place, | yes, nearer one, since the wave is flatter there. one point five seven is the flattest place we have tried, so I put the slope near one. the slope reads 0.9, so the wave is nearly level at one. so the slope changes from place to place, and that changing number is the derivative. what will the slope read at minus one? |
| B | one holds, and the numbers show why. at one, doubled sine is about one point six eight, so I put the wave just under its top. near one, I make the wave climb about one for each one along. the slope reads one point zero eight, so near one the wave climbs about one per step. so the slope changes from place to place, and that changing number is the derivative. | one holds, and the wave near one shows why. doubled sine of one is one, so I put the wave at minus one and a half there. the slope reads minus one and a half, so the wave falls there as steeply as it rose at zero. so the slope changes from place to place, and that changing number is the derivative. what will the slope read at minus one? |

### dlg-00005-074 turn 0

**Pupil:** a curve has just one slope all the way along?

**Tape tutor:** worth testing rather than trusting, so let's work numbers on this cubic first. if one steepness held everywhere, I'd expect our curve to be straight. at one, one cubed is one, three times one is three, and one take away three is minus two, so I'm betting our curve sits at minus two above one. if the input nudges a bit past one, how many times that nudge will the height move?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | it's a natural guess, since a straight line has one slope everywhere, so let's test it on a curve. I expect this curve to climb at first, then turn over and fall, since it doubles a sine. at zero, sine of zero is zero, and twice zero is still zero, so I hold the curve sits at height zero there. if the input moves to one tenth, how far will the output move? | it does, and that one slope is why a hump stands out. a hump is a place where the curve is highest, so we should find it steepest there. at zero, the cube is zero, and taking three from it leaves minus three, so we have the curve sitting three below the axis there. if the input moves from zero to zero point one, how much will the height drop? |
| A | it does, and that slope is what tells us how fast the height changes. if the height changes slowly, we should find the curve flat where it crosses the middle. at two, two times two is four, and one taken from that leaves three. so I hold that the curve stands three above the axis at two. if I nudge the input to two point one, how far will the height move? | (silence) |
| B | it does, because a wave is only steep where its height changes fast, and this wave climbs only once. at minus two, I expect the wave to sit below the axis, since two cubed less three twos is minus two. if I nudge the input to minus one, how far will the output move? | (silence) |

### dlg-00005-074 turn 2

**Pupil:** up only slightly, since the curve bottoms out near there.

**Tape tutor:** yes, only slightly, and you're right that the dip is there. we should find it barely above minus two at one point one, since one point one cubed is one point three three one, less three point three. a tenth of a step lifted it three hundredths, so I'd say the curve is flat right at one. zero slope, so at the bottom it neither climbs nor falls. at two, what will the slope read?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | a fair guess, so let's try it. if your guess holds, I'd expect the curve to rise only a tiny bit past one, since it falls away to zero there. | yes, a small step should show how little the curve moves there. one point one cubed is one point three three one, less three point three, so I put the curve at minus two point six one. a tenth past one, I'd bet it climbs near a tenth for each tenth along. the slope is near zero at the bottom, so the curve is level there. at zero, how much will the slope read? |
| A | right, only slightly, so let's test your call. if your rule holds, I'd expect the curve to rise just under one for one point one, since one point one cubed is one point three three one, less three point three. the height went from minus two to minus one point nine seven, so the curve climbed about once the step. at zero, where the curve crosses the axis, how many times the step will the height move? | right, and since the curve is level at one, I'm betting its steepness at one is zero. the slope is zero, so the curve is flat at one, though its height is minus two. at zero, where the curve is flat too, how many times the step will the height move? |
| B | right, and that little more than one cubed is what we need to turn a drop of one into a climb of many times that nudge. one point one cubed is one point three three one, so we have the curve climbing about three for each one across at one. the slope is three, so the climb is three times the step. at zero, where the curve crosses the middle, how many times the step will the climb be? | right, and the slope is zero at one, so the curve is level there. so I'm betting the curve at one sits level with the ground there. |

### dlg-00005-074 turn 4

**Pupil:** zero, since a curve keeps one slope all the way along.

**Tape tutor:** a natural thing to think, since at one the slope does read zero. by your reckoning, we ought to find the curve as level at two as at one. the slope climbed to nine, so near two the height moves nine times any small step. so at zero, between the hump and the dip, how many times the step will the height move?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, zero, because the bottom has no climb left. two cubed is eight, less six, so I'm betting our curve sits at minus two there. a tenth past two, the | yes, zero, since the curve is level at the bottom. two cubed is eight, less three is five, so I'm betting our curve sits at minus five there. a tenth of a step lifted |
| A | yes, zero, because the curve keeps one slope all along. two cubed is eight, less three, so I'm betting our curve sits at minus two there. if the input | yes, zero, since the curve is level at two. two cubed is eight, less three is five, so I'm betting our curve sits at minus two there. a tenth of a step lifted it two tenths, so I'd say the curve climbs two for every one along at two. the slope is two, so the curve climbs two for every one along at two. at three, how much will the slope read? |
| B | yes, zero, because the curve lies flat at its bottom and climbs out of it. two cubed is eight, less three, so I'm betting our curve stands two above the axis at two. | yes, zero, since the curve is level at two. two cubed is eight, less three is five, so I put our curve at minus two there. a tenth of a step lifted it two tenths, so I'd say it climbs two for every one along at two. the slope went from minus two to two, so the curve climbs two for every one along at two. |

### dlg-00005-074 turn 6

**Pupil:** down, because the curve falls between the hump and the dip.

**Tape tutor:** right, it falls, and at zero the height is zero. one tenth cubed is a thousandth, less three tenths, so I make our curve nearly three tenths under zero at one tenth. a drop of three tenths over one tenth, so I'd put the fall at zero near three times the step. minus three, so here the height drops three times the step. at one half, how many times the step will the height fall?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, down, because past the dip the curve heads toward the bottom. I'd put the curve at zero below minus two, since zero cubed is zero, and taking away three makes minus three. so I expect it to climb nine times the step at zero, as it did near two. the slope ran from minus three to nine, so the curve climbs nine for each one across there. at minus one and a half, how many times the step | yes, down, because the curve falls between the hump and the dip. I'd put its fall near zero at zero, since the hump is the highest point. the slope reads minus one point nine nine, so the curve drops one point nine nine per step at zero. so at minus one and a half, how many times the step will the height move? |
| A | right, it falls, and between the hump and the dip it falls just as hard. zero cubed is zero, and three times zero is zero, so I'm betting our curve crosses the axis at zero. if you're right, it must drop from its hump to zero with no change in steepness. the slope ran from nine down to minus two, so the curve falls twice as hard between the hump and the dip as it climbs between one and two. at minus one and a half, | yes, down, and the steepness is the number of times the step the height moves. at zero, I'd put the curve falling nine times as hard as it climbs near two. the slope ran from nine down to minus nine, so the curve falls nine per step at zero, where it stands nine high. at minus one and a half, how many times the step will the height move? |
| B | yes, two, because the curve falls between the hump and the dip. at zero, the cube of zero is zero, and three times zero is zero, so I'm betting our curve sits at height zero there. if you're right, I'd expect it to drop two for every one across at zero. the slope ran from nine down to minus two, so the curve drops two for every one across there. so at minus one and a half, how many times the step will the height move? | yes, about zero, since the curve falls between the hump and the dip. zero cubed is zero, less three is minus two, so I'm betting the curve lies below the axis there. a tenth of a step down took one tenth of a unit off the height, so I take the curve to fall one per step there. the slope is zero, so the curve lies level at zero. at one and a half, how many times the step will the height move? |

### dlg-00005-074 turn 8

**Pupil:** three, since the slope at zero was minus three.

**Tape tutor:** a fair guess, since zero gave three. going by your three, I'd make our curve drop as steeply at one half as at zero. instead it falls two and a quarter times the step there. nine at two, zero at one, minus three at zero, so the steepness changes from place to place, and that changing number is the derivative. so did the curve fall more gently at one half than at zero?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, three, since the slope at zero was minus three. one half cubed is a quarter, less three quarters, so I make our curve below zero at one half. three tenths down for one tenth of step, so I'd say it falls three times the step there. the slope ran from minus three to minus three and a half, so the fall changes from place to place. at minus two and a half, how many times the step will the height fall? | yes, minus three, and the fall is steeper here than at zero. one and a half cubed is two and a quarter, less three and a half, so I make our curve below zero at one and a half. so the slope reads minus three, steeper than the fall of three tenths per tenth at zero. at minus one and a half, how many times the step will the height fall? |
| A | yes, three, and your three at zero is the slope we should find at one half too. one half cubed is a quarter, less three quarters, so I make the curve below zero there. three tenths down over one tenth, so I'd say it falls three times the step there. the slope ran from minus three to minus three and a half, so the fall changes from place to place. at minus two and a half, how many times the step will the height fall? | yes, minus three, and the fall is steeper here than at zero. one and a half cubed is two and a quarter, less three and a half, so I make our curve below zero at one and a half. the slope is three, so the curve drops three times the step here. at minus two, how many times the step will the height fall? |
| B | right, minus three, and the slope at one half is minus two and a half, so the curve falls harder there. so I'd say the curve at one half drops two and a half times the step, just as the slope said. the slope ran from minus three to minus two and a half, so the steepness changes from place to place. at minus one and a half, how many times the step will the height fall? | yes, minus three, since the curve lies below the axis at one and a half. the slope fell from nine to minus three, so the curve drops three times the step there. so at one and a half the curve drops three times the step, while at zero it falls three times the step. at two, how many times the step will the height fall? |

### dlg-00005-081 turn 0

**Pupil:** so, how would i work out a derivative? i'm not sure.

**Tape tutor:** you work one out from numbers, one input at a time, so I'll start with a wave. we know this wave never climbs past two, because a sine tops out at one and we double it. at one, the sine is about point eight four, so we have the wave sitting about one point six eight high there. if I nudge the input to one point one, how much do you think the height climbs?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | let's work numbers before naming them, because a derivative is about how fast something changes. this wave doubles its sine, so we should find it climbing and falling. at input one, the sine of one is about point eight four, so we should find the wave there sitting about two point five six high. if the input moves to one point one, how far will the height move? | it's a natural thing to wonder, so let's test it with numbers. we know a cube less three cubed over eight is zero, so we should find this curve crossing the axis at one. one cubed less three is minus two, so we should find the curve two below the axis there. if the input moves to one point one, how far will the height move? |
| A | it's a natural thing to wonder, because derivatives sound like numbers you find on a curve. so let's work one out on numbers before trusting a rule. we expect this curve to climb to a hump, then fall to a dip. at input zero, the cube is zero, and three times zero is still zero, so we have the curve crossing the axis there. if the input moves from zero to a tenth, how far do you guess the height will move? | (silence) |
| B | you can work out a derivative with numbers, so let's try it. I expect this curve to climb and fall, since a cube outgrows a cube. at input one, one cubed is one, and one minus three is minus two, so I say the curve sits at height minus two there. if the input moves to one point one, how far will the height move? | (silence) |

### dlg-00005-081 turn 2

**Pupil:** it climbs to about one point seven eight, since sine still rises.

**Tape tutor:** right, it climbs to about one point seven eight. we should find it about point one above the old height, because the sine still rises there. so a step of point one lifting it point one tells me the wave at one climbs about one per one along. the slope reads just over one, so the climb matches the step. at zero, where the wave has no height, how much will it climb per step?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, about one point seven eight, since sine still climbs. so I hold the wave must be near its top at one point one, since it barely climbs there. the slope is zero point two eight, so the wave climbs two point eight times the step. at the input zero, where the height is zero, how much do you think the slope reads? | yes, about one point seven eight, since the sine is still climbing there. so we have the wave climbing about one per step along at one point one. the slope is one, so the wave climbs one for each step along at one. at zero, how much will the slope read? |
| A | yes, about one point seven eight, and the wave itself shows it. we should find the wave at one point one about one point six nine high, since the sine there is about point one nine four. so the second height rose about as much as the step, and I take it the wave climbs about one per one here. the slope is one, so the wave climbs one for one at one. at zero, | yes, about one point seven eight, and the wave itself settles that. we have it climbing about one per step at one, so I take the wave to be level at one point one. the slope is zero, so the wave is level at its top. at zero, how much will the slope read? |
| B | yes, about one point seven eight, and the wave itself shows that. we get about point eight four from the sine of one point one, so we have the wave climbing about one per step there. so the slope is one, and it matches our one point six eight high. at the input zero, where the height is zero, how much do you think the slope reads? | yes, about one point seven eight, and the sine is still the best guess for how steep it is. the wave is one point six eight high at one, so I take it to climb about one per step along there. the slope is one, so the wave climbs one for each step along at one. at zero, where the height is zero, how much will the slope read? |

### dlg-00005-081 turn 4

**Pupil:** less than at one, since the wave sits lower there.

**Tape tutor:** let's try zero. two sine of zero is zero, and two sine of point one is about point two. so I'd say the wave at zero climbs twice its step, steeper than at one, though it sits lower. the slope rose to two by zero, so the wave climbs fastest where its height is nothing. at the peak, one point five seven, how much will it climb per step?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, less than one, because the wave is lower there. at zero the sine is zero, so we have the wave sitting at ground level. if you nudge the input to zero point one, how much will the slope read? | right, less than one, because the wave is flat at zero. we have it level at zero, since the sine of zero is zero, doubled to one. so the slope fell from one to zero, as the wave flattened. the slope fell to zero at the wave's lowest point, so the steepness changes from place to place. at two, how much will the slope read? |
| A | right, less than one. at zero the wave sits at height zero, so we have it climbing nothing per step there. so the slope fell from one to nothing, and the wave's height there is zero, so the two are separate numbers. at the peak, two point five seven, how much will the slope read? | yes, less than one, because the wave has no height at zero. we should find it climbing about zero point nine per step there. the slope fell from one to zero point nine, so the wave climbs less per step as it moves from one to zero. at two, how much will the slope read? |
| B | right, less than one, because the wave lies flat there. we should find it barely climbs at zero, since the sine there is near zero. so the slope fell from one to about zero, and the flat wave at zero is the same height as the steep one at one. at two, how much will | yes, less than one, because the wave is flat at zero. we have it level at the crossing, since the sine of zero is zero. so the slope fell from one to zero, so the wave is level at its crossing. at two, how much will the slope read? |

### dlg-00005-081 turn 6

**Pupil:** neither up nor down, since the top of the wave is flat.

**Tape tutor:** right, the top is flat, so it neither climbs nor falls there. two sine of one point five seven is two, so I doubt any point on this wave stands higher. and zero climbed steepest with no height, so I think the climb is all gone up here at the most height. the slope fell from two to zero, so height and steepness differ. at two, just past the top, how much will it fall per step?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, flat. one point five seven sine is about one point five seven, so I put the wave's top at height zero. if you nudge the input to one point six, how much will the slope read? | right, flat. the peak is where the wave has no height at all, so I'd put its steepness at zero. the slope fell to zero while the height rose to one point five seven, so the wave climbs nothing per step there. the slope fell to zero while the height rose to one point five seven, so the wave climbs nothing per step at its top. at minus one point five seven, how much will it climb per step? |
| A | right, flat. one point five seven sine is about one point five seven, so I take it the wave stops climbing there. so I claim the wave climbs nothing per step at its top, though its height there is one. the slope fell from two to zero, so the steepest climb came at the lowest place. at four point five, how much will the slope read? | right, it falls there, since the wave has stopped climbing at the peak. one point five seven sine is about one point six eight, so we have it falling about one per step there. the slope fell from two to minus one, so the wave falls hardest where it stands highest. at minus one point five, how much will it climb per step? |
| B | right, flat. one point five seven sine is about one point five seven, so I take the wave to sit there at height one. if you nudge the input to one point six, how much will the slope read? | right, flat, and the slope fell from two to zero. so we have the wave at its peak climbing nothing per step, though its height is one point five seven. the slope fell to zero, so the wave climbs nothing per step at its top. at minus one point five seven, how much will it climb per step? |

### dlg-00005-081 turn 8

**Pupil:** up, since the wave still stands high there.

**Tape tutor:** let's try two, which is high. I get two sine of two as one point eight two, and two sine of two point one as one point seven three, so we drop. that's point oh nine down per step, so I'd put the fall at two just under one per one along. the slope reads minus point eight three, so a high wave still falls. at two, is the wave climbing or falling?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, it still stands high there, so I expect it to fall. two point five sine is about one point nine nine, so I make the wave below the top there. and since it still stands high, I hold it falls there more gently than at the peak. the slope fell to minus one point nine nine, so the wave falls gently there while still high. at three point five, how much will it fall per step? | right, it falls there, since the wave has stopped climbing. two sine of two is zero, so I put the wave at ground level there. and since it has stopped climbing, I hold it falls there. the slope fell from zero to minus one, so the wave falls one per step there. at three, how much will it fall per step? |
| A | right, it still stands high there, so the fall must be gentle. two point five seven sine is about one point six eight, so I put the wave below the top by that much. and since it falls gently, I take the fall to be about one per one along. the slope fell to about one, so the fall is gentle there while the wave stands high. at three, how much will it fall per step? | right, it still stands high, so I'd say it falls more gently than at zero. two sine of two is zero, so I doubt the wave falls below zero there. and the slope fell from zero to minus one, so the wave falls one per step there. at three, how much will it fall per step? |
| B | right, it still stands high there, so the fall must come from its turn away from the top. two point five seven is about two point one five, so I put the wave below the top there. and I'd bet it falls there faster than at zero. the slope fell to minus one point three one, so the wave drops one and a third per step there. at three point four six, how much will it fall per step? | right, it still stands high there, so I'd say it falls more gently than at zero. two sine of two is two, so I put the wave falling two per step there. the slope fell from zero to minus two, so the wave falls two per step at two, though it stands two high. at three, how much will it fall per step? |

### dlg-00005-087 turn 0

**Pupil:** sorry, how would I work out a derivative of a curve myself?

**Tape tutor:** no need for sorry, you work it out with arithmetic. the height here is the input cubed less three times it, so any height can be worked by hand. I doubt such a curve stays level anywhere. at two, two cubed is eight and three twos are six, so I expect eight less six. that leaves two, so I hold the curve stands two high there. at two point one, how high do you guess?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | you work it out from a second height, because a derivative is how fast the height changes. I expect this curve to climb, since the cube outgrows the square. | (silence) |
| A | you work out a derivative yourself, so let's find one on numbers instead of rules. I expect this curve to climb and fall, since a cube and a half cube both grow faster than one would guess. at input one, I get a cube of one and a half cube of one as one and a half, so I hold the curve stands one and a half high there. if the input moves to one point one, how far will the height drop? | (silence) |
| B | you work out a curve by feeding numbers to an equation and seeing how far the output moves for a small step. so we should find this cubic climbing to a hump and then falling to a dip. at zero, the cube is zero, and three times zero is zero, so we have the curve crossing the axis there. if the input moves from zero to zero point one, how far will the output move? | it's a natural thing to want a rule, so let's test one. if a curve follows a rule, we should find it never climbing higher than its input. numbers first, because numbers never lie. zero cubed is zero, and three times zero is zero, so we have the curve crossing the axis at zero. if the input moves from zero to one, how far will the output move? |

### dlg-00005-087 turn 2

**Pupil:** two point nine six one, because the cube grows fast.

**Tape tutor:** yes, that is right, because the cube climbs quickly once we are past two. we should find the curve nearly a whole unit higher, just a tenth of a step past two. the height now reads two point nine six, so a tenth of a step lifted the curve almost ten times as far. at nought, where the curve sits at height nought, how many times the step do you guess the height moves?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, two point nine six one, because the cube grows fast. so I hold the curve climbs past two to two point nine six one, since two point one cubed is three point four six one and three point three two cubed is three point nine six one. that tenth of a step lifted it nearly a whole unit, so I take the curve at two to climb about nine per step. the slope is nine, so the curve climbs nine for every one across. at zero, | yes, it grows fast, and that growth is the derivative. two point one cubed is one point three three one, less three point three, so I hold the curve sits just under two there. the slope is zero, so the curve lies level at two. at two point two, how much will the slope read? |
| A | right, it climbs fast there, so I hold the curve stands just under two point nine six one at two point one. the slope is nine point six one, nine point six one up for one along. at two, the top, how much do you guess the slope reads? | yes, it grows fast, and the height is the one we should test. two point one cubed is six point one six one, less three point two, so I hold the curve stands two point nine six one there. so the slope is two, twice the step. at zero, how steep do you guess it is? |
| B | right, it climbs fast there, so I take your point to be the top sits just past two. the height went from two to two point nine six one, so the curve climbs nine point six one times the step. that steepness is the derivative, and I hold the curve climbs harder at two than at two. the slope is nine point six one, while the height is two. at three, how steep do you guess the curve climbs? | yes, it grows fast, and that growth is what we test at two point one. if the curve were level at two, I'd expect it to climb only a tenth over a tenth. the slope is one, so the curve climbs one for each step along at two. at zero, how steep do you guess it climbs? |

### dlg-00005-087 turn 4

**Pupil:** zero point one, since a low curve should be fairly flat.

**Tape tutor:** a fair guess, so let us try it a tenth past nought. I am betting the curve drops there, because a tenth cubed is only a thousandth while three tenths come off. so we must land nearly three tenths below where the curve started. the second height sits three tenths below nought, three times the step, so low is not flat. at one, in the dip, how many times the step will the height move?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, about one tenth, since the curve at nought barely climbs. so I hold the curve at nought moves about one tenth for each tenth we step along. the slope is one point zero eight, so the curve climbs nearly one for each step there. at one, how many times the step do you guess the height moves? | yes, zero point one, because the curve is level at nought. a tenth past nought, I hold the curve sits just under its top at nought point one. so I make the curve climb about one per step there, just under its steepness at two. the slope is zero at the top, yet the curve climbs there just under one per step. at minus one and a half, how many times the step do you guess the height moves? |
| A | yes, zero point one is right, because a flat curve barely moves with its place. at nought, the cube is zero, and three times zero is still zero, so I hold the curve sits at nought there. a tenth on, at nought point one, I expect it to rise just under | yes, zero point one, because the curve is level at its lowest point. the slope is zero, so the curve lies level at its lowest height. at one, how many times the step do you guess the height moves? |
| B | yes, zero point one is right, because a low curve barely moves. we should find the curve at nought barely above nought, since a tenth of a step lifted it only a hundredth. the height went from two point nine six to nought point one six, so about one tenth of a step lifted one tenth of the height. at one, how many times the step do you guess the height moves? | yes, zero point one, because the curve is level at nought. a tenth of a step lifts the curve only a tenth, so I hold it level there. the slope is zero, so the curve is level at its lowest point. at one, how many times the step do you guess the height moves? |

### dlg-00005-087 turn 6

**Pupil:** less than one time the step, since the dip is nearly level.

**Tape tutor:** yes, well under once, because the dip is nearly level. I make one point one cubed one point three three one, so taking three point three off leaves the curve just above minus two. so a tenth of a step barely lifts it, and right at one I hold the curve dead level. the slope is zero, flat at the bottom of the dip. at two, how much will the slope read?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, less than one time the step, because the dip is nearly level. one cubed is one, and three times one is three, so I hold the curve sits at nought there. a tenth on, it is one point one cubed less three point one, so I claim the curve climbs there. the slope is one point one against nought, so low is not | yes, less than one is right, because the dip is nearly level at one. we should find the curve nearly level at one, since the cube is nearly one and three take away one is minus two. the slope is zero, so the curve lies level at one, though its height is minus two. at minus one and a half, how many times the step will the height move? |
| A | yes, one and a half times the step is right, because the dip is nearly level. we should find the curve barely climb out of its dip at one, since its climb from nought has barely started. the slope is one and a half, so the curve moves that many times the step there, while at nought it moved three times the step. at minus one and a half, how many times the step will the height move?; | yes, one and a half times the step is right, because the curve falls steeply there. so I hold the curve drops one and a half times the step at one, though it stands nearly level at nought. the slope is minus one and a half, so the curve drops one and a half per step there, though it stands nearly level at nought. at minus one and a half, how many times the step will the |
| B | yes, less than one is right, because the dip is nearly level. one cubed is one, and three times one is three, so I hold the curve sits at nought there. if the step were a tenth bigger, how many times the step would the height move? | yes, less than one is right, because the dip is nearly level. we should find the curve nearly level at one, since a tenth cubed is a thousandth and three tenths off. so I hold the curve nearly level at one, though its height there is minus one point nine nine. the slope is zero, so the curve is level at its lowest point. at minus one and a half, how many times the step will the height move? |

### dlg-00005-087 turn 8

**Pupil:** two, since the curve stands two high there.

**Tape tutor:** fair guess, two is the height there, so let us try it. if height sets steepness, I expect the curve at two to climb only twice its step. the slope says nine, so the curve climbs nine times its step there, though it stands only two high. that number differs from place to place, and that changing number is the derivative. so what gave us the nine at two, the height or the climb?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, two, because two cubed less three twos is eight less six, so the curve stands two high there. so I hold it climbs two for every one across at two, its steepness. the slope reads two, so the curve climbs two for every one across at two, while at one it was zero. at nought, how much will the slope read? | yes, two, because the curve stands two high there. two cubed is eight, and three twos are six, so I make the curve two high there. so the slope is two, two up for every one step along. the slope is two, so the curve climbs two for every one step along at two. at minus one and a half, how much will the slope read? |
| A | yes, two, because two cubed is eight, less six, so we find the curve two high there. and since two cubed over eight is two, I hold the curve climbs twice as fast as it goes out. the slope ran from zero to two, so the curve climbs two for every one across at two. at the peak, one point five seven, how much will the slope read? | yes, two, because the curve climbs steeply there. two cubed is eight, and three twos are six, so I make the curve sit two high there. so I hold it climbs two for every one along at two. the slope is two, so the curve climbs two for every one along at two. at minus one and a half, how much will the slope read? |
| B | yes, two, because two cubed less three twos is minus one point three three, so I expect the curve to climb out of its dip there. the slope went from zero to nine, so the curve climbs nine times the step at two, while at one it barely moved. at minus one and a half, how much will the slope read? | yes, two, because the curve stands two high there. two cubed is eight, and three twos are six, so I make the curve two high there. so the slope is two, two times the step, and that number changes from place to place. at one and a half, how much will the slope read? |

### dlg-00005-091 turn 0

**Pupil:** hi, how would I find a curve's slope at one point?

**Tape tutor:** good question, so let's find out with numbers. since one minus a square never climbs past one, we should get a hill. at one, one squared is one, and one minus one is zero, so we have the hill at ground level there. a tenth further, one point one squared is one point two one, so we must already be below ground. so at one, where the height is zero, how steep is the hill?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | good question, so let's work one out on numbers. I expect this cubic to climb fastest where its hump is highest, since a hump is what a climb follows. at minus two, cubing gives minus two, and taking away three times minus two leaves two, so I hold the curve sits at height two there. if I nudge the input to minus one, how far will the height drop? | it's a natural thing to want a number that never changes, so let's |
| A | you find a curve's slope at one point by nudging the input a tiny bit and seeing how far the output moves, so let's test that on this wave. I expect this wave to climb to a hump, then fall back to zero. at zero, the sine of zero is zero, and twice zero is still zero, so I hold the wave sits at height zero there. if the input moves from zero to zero point one, how far will the output move? | (silence) |
| B | you find it from an input number, because a curve is how a number gets turned into a height. we expect this curve to climb to a hump, then fall back to zero. at zero, the cube is zero, and three times zero is still zero, so we have the curve sitting at height zero. if the input moves to one, how far will the height move? | (silence) |

### dlg-00005-091 turn 2

**Pupil:** zero, because the height there is zero, so it's flat.

**Tape tutor:** you call it flat since the height is zero, so let's try it. the height fell zero point two one over a step of one tenth, two point one times the step, so I suspect the hill at one is anything but level. the slope is minus two, so the hill drops two for each step across, even at zero height. at x minus one, also at zero height, how steep is the hill?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, zero, because the hill is level at one. so I hold that the hill must lie flat at one, with no climb or fall there. the slope is zero, so the hill is level at one, though its height is zero. at two, how much will the slope read? | yes, zero, because the hill is level at one. the slope is zero, so the hill is level at one too. the slope is zero, so the hill is level at one, though its height is zero. at two, how steep do you guess the hill is? |
| A | right, it is flat there, because the hill is level at one. so I hold that the hill must be falling just as hard as it climbed there, since one point one squared is one point two one. the slope is minus one, so the hill drops one for each step along at one. at two, where the height is zero again, how steep do you guess the hill is? | yes, zero height and zero steepness match, so let's test your claim at one. if your rule holds, the hill should lie flat at one. |
| B | yes, flat, and the numbers show why. a tenth along lifted the hill a tenth, so we expect it to climb about one for one at one. the slope is zero at ground level, so the hill is level there though its height is zero. at two, how much will the slope read? | yes, flat, and the numbers back it up. one point one squared is one point two one, so we have the hill two hundredths below ground there. so I claim the hill is level at one, though its height is zero. the slope is zero, so the hill is level at ground level. at two, how much will the slope read? |

### dlg-00005-091 turn 4

**Pupil:** two, going up, since the hill climbs toward its top there.

**Tape tutor:** yes, two, and climbing, because that side still rises toward the top. we have a hill mirrored across x zero, so its climb at minus one must match its drop at one, sign turned over. the slope went from minus two, through zero on the peak, to two at minus one, so steepness changes from place to place. out at x minus two, how steep is the hill?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, two, because the hill climbs toward its top at minus one. minus one squared is one, and one minus one is zero, so we have the hill back at ground level there. a tenth on, minus one point one squared is one point two one, so we must be near its top. so I take it the hill climbs two for each step across there, its steepest place yet. the slope went | yes, two, because the hill climbs two for each step across there. the slope went from minus two to two, so the hill climbs two for each step across at minus one, just as you said. the slope is two at minus one, so the hill climbs two for each step across there. at x zero, the very top, how steep is the hill? |
| A | yes, two, because the hill climbs toward its top. minus one squared is one, and one minus one is zero, so we have the hill back at ground level there. two point one squared is four point two one, so we must be near its top by minus one. so I take it the hill climbs two for each step across there, just as you said. | yes, two, because the hill climbs two for each step across there. the slope went from minus two to two, so the hill climbs two for each step across at minus one, just as you said. the slope is two at minus one, so the hill climbs two for each step across there, though its height is zero. at x zero, the very top, how steep is the hill? |
| B | yes, up, because the hill climbs toward its top at zero. minus one squared is one, and one minus one is zero, so we have the hill back at ground level there. a tenth on, minus one point one squared is one point two one, so we must be near its top. so I take it the hill climbs three for each step across there, its steepest place yet. the slope reached three, | yes, it climbs, since the hill is past its lowest point. one point one squared is one point two one, less one, so we have the hill two point one above ground there. so I expect it to climb about one per step across there. the slope is one, so the hill climbs one for each step across, just as you said. |

### dlg-00005-091 turn 6

**Pupil:** four, going up, since the slope doubles as x doubles.

**Tape tutor:** yes, four, since the steepness runs at twice how far out we stand. I make the height minus three at minus two, and minus two point six one a tenth inward. that rise of zero point three nine is three point nine times the step, so to my mind the hill climbs four right at minus two. it climbs four there, twice its pace at minus one. at x one half, how steep is the hill?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, four, and up, because the slope doubled as x doubled. two squared | yes, four, and doubling as x doubles is a fair guess. so I hold the hill at minus two climbs four for each step across. the slope went from two to four, so the hill steepens as it moves out. at x minus three, how steep is the hill? |
| A | yes, four, and doubling the step is the right rule. two times the step gave two, so we expect the hill at minus two to climb four for each step across. the slope went from two at minus one, to zero on the top, to four at minus two, so steepness changes from place to place. at x zero, how steep is the hill? | yes, four, and doubling holds. the hill climbs four for each step across at minus two, so I take it to be steepest of all. the slope went from two at minus one, through zero on the top, to four at minus two, so steepness changes from place to place. at x zero, how steep is the hill?. |
| B | yes, four, and doubling holds because the hill grew steeper as we went from one to minus one. so I expect the hill at minus two to climb four for each step across. the slope went from two back to zero, so the hill climbs four at minus two, where its height is minus two. at x zero, the very top, how steep is the hill? | yes, four, and the hill climbs four for each step across there. so the hill climbs four for each step across at minus two, just as steep as at minus one. the slope went from two to four, so steepness changes from place to place, and that changing number is the derivative. what does the derivative tell you that the height cannot? |

### dlg-00005-091 turn 8

**Pupil:** minus one, because the slope is minus twice x.

**Tape tutor:** yes, minus one, as twice a half is one. a half squared is a quarter, so I put the hill three quarters high at one half. it stands above ground, yet I claim it falls one per step there. so the derivative changes place to place, whatever the height. would lifting the hill change that slope?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, minus one, because twice x is the rule we have found. so I hold the hill at one half drops one for each step across, and falls harder than at minus two. the slope fell from four to minus one, so the hill steepens as it goes out. at x two, how steep is the hill? | yes, minus one, because the slope is minus two there, twice the step. so I hold the hill falls one for each step across at one and a half. |
| A | yes, minus one, because twice x is the rule we have found. so I hold the hill at one half drops one for each step across, and falls one for each step at x one. the slope fell from four to minus one, so the hill drops one for each step there, its own pace. at x two, how steep is the hill? | yes, minus one, because the slope is minus two times x, and at one half the hill drops one and a half per step. so I take it to be level at one half, where the hill falls one and a half per step. the slope ran from minus two to zero, so the hill is level at one half. at x zero, how steep is the hill? |
| B | yes, minus twice x, and that rule is the derivative. I make the hill at one half one point three nine above ground, so its climb there is just over two. the slope fell from four to just over two, so the hill steepens as it goes out. at x two, how steep is the hill? | yes, minus twice x, because the hill climbs four at minus two and falls four at one and a half. the slope fell from four to minus two, so the hill drops two for each step across there. so the slope changes from place to place, and that changing number is the derivative. does the hill at x one and a half climb or fall? |

### dlg-00005-094 turn 0

**Pupil:** so, if I had a curve, how could I find its steepness myself?

**Tape tutor:** you nudge an input and watch how far the output moves, so we'll try it on two times sine x. sine never passes one, so I expect this wave to stay within two either way. at one, sine is about point eight four, so doubled I claim the wave stands one point six eight high there. stepping from one to one point one, how far will the height climb?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | you can find it yourself with numbers, because a curve's steepness at a point is how fast its height changes as you move a small step along. I expect this curve to climb, since the cube outgrows the square past one. at one, one times one is one, and three times one is three, so I hold the curve sits at height three there. if I step the input from one to one point one, | let's find that with numbers, because numbers never lie. we know a hump must sit below the axis at its top, since it has to lose height all the way up. at zero, the cube is zero, and taking three zeros away leaves zero, so we have the hump sitting at height zero there. if the input moves to one, how far will the hump drop? |
| A | you can find a curve's steepness yourself with numbers, so let's work one out. I expect this curve to climb fastest where its hump is highest, since a hump is where the climb is steepest. at input two, I get the curve standing two high, since two cubed less three twos is minus one, plus two. so I hold that the curve stands two high at input two, and climbs toward | (silence) |
| B | you can find a curve's steepness at a place with numbers, because a steep curve climbs fast where it rises fast. I expect this cubic to climb fastest where its hump stands highest, since a hump is the steepest place. at minus two, cubing gives minus two, and taking two away leaves two, so I hold the curve sits at height two there. if the input moves to minus one, how much | it's a natural thing to want a number that never changes, so let's test one against numbers. if a hump never moves, then we should find its height at zero never changes. at zero, the cube is zero, and taking three zeros away leaves zero, so we have the hump as high as its height at zero. at one, the cube is one, and three ones away leave two, so we have |

### dlg-00005-094 turn 2

**Pupil:** up about zero point one, since the slope there is near one.

**Tape tutor:** right, a tenth, just as you said. I reckon two sine of one point one comes to one point seven eight, a tenth higher than at one. so a tenth of climb for a tenth of step is one rise per step. my guess is the curve right at one climbs just about that steeply too. and the slope agrees, just over one. so at zero, where the height is zero, how much will the slope read?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, about zero point one, since near one the wave still climbs nearly one for each step along. so I hold that the wave must stand about one point zero eight at one point one, a tenth higher. the slope is zero point one zero, so the wave climbs nearly one for each step along there. at the peak, x zero, how much will the slope read? | yes, about point one, since the wave is near its top at one point one. so I hold that the wave climbs about one for each step along at one. the slope is one, so the wave climbs one for each step along at one. at zero, where the wave is flat, how much will the slope read? |
| A | yes, about zero point one, since the wave barely climbs there. so I hold that the wave barely rises from one to one point one, since it barely moves there. the slope is zero there, while the height is one point six eight. at zero, where the wave crosses the middle, how much will the slope read? | yes, about point one, since the wave barely climbs past one there. so I hold that the wave climbs about one for each step along at one. the slope is one, so the wave climbs one for each step along at one. at zero, how much will the slope read? |
| B | yes, about zero point one, since the wave barely climbs there. so I hold that the wave barely climbs at one, since its steepness is nearly zero. the slope is zero, so the wave climbs nothing there, though its height is one point six eight. at zero, where the height is zero, how much will the slope read? | yes, about zero point one, since the wave barely climbs there. so I hold that the wave climbs about one for each step along at one. the slope is one, so the wave climbs one for each step along at one. at zero, where the wave is flat, how much will the slope read? |

### dlg-00005-094 turn 4

**Pupil:** two, since the wave climbs steepest as it crosses zero.

**Tape tutor:** yes, two, the wave crosses the middle at full tilt. and I get two sine of point one at point two, so the curve rises twice its step. I count the wave nearly twice as steep at zero as at one. and the slope reaches two at the crossing, with no height at all. so at the peak, one point five seven, how much will the slope read?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, two, because the wave crosses zero upwards at zero. I take it to be a flat wave at zero, since its height there is zero. the slope went from one to two, so the wave climbs twice as hard at zero as at one. so the slope changes from place to place, and that changing number is the derivative. at three point five seven, how much will the slope read? | right, two, because the wave is flattest at zero, where its height is zero. so I claim the wave climbs two for every one along at zero. the slope went from one to two, so the wave steepens as it moves from one to zero. so at minus one, how much will the slope read? |
| A | right, two, because the wave doubles sine there, and sine of zero is still zero. so I claim the wave crosses the middle at zero with no height at all, yet climbing two for every one along. the slope went from one to two, so the wave climbs twice as hard at zero as at one. at three point five seven, how much will the slope read? | yes, two, because the wave is flat at zero and steepest at one. so I claim the wave climbs two for every one along at zero. the slope is two, just as we expected. so at minus one and a half, how much will the slope read? |
| B | right, zero height and steepest climb go together. sine of zero is zero, so I claim the wave crosses the middle there at its steepest. so I make the wave climb about one per step there, just as your rule says. the slope came out one there, where the height is zero. at three point one four, how much will the slope read? | right, zero, because the wave has nowhere to go there. zero sine is a height of zero, so I claim the wave must lie flat at zero. the slope went from one to zero, so the wave steepens as it crosses the crossing. so at minus one, how much will the slope read? |

### dlg-00005-094 turn 6

**Pupil:** zero, since the wave stops climbing right at the top.

**Tape tutor:** yes, zero, the top is where climbing turns to falling. two sine of one point six seven is one point nine nine, so a tenth past the peak, I'm betting the wave has sunk just a hundredth. so to my mind, steepness drains from two at the crossing to nothing up top. the slope ran from two down to zero, flat on the peak. so at two, past the top, how much will it read?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, zero, because the climb stops at the top. I make the wave one point six eight high at one point five seven, so its climb is gone. so the slope fell from two to zero, while the height rose from zero to one point six eight. so the steepness followed the height, not the place on the board. at three point one four, how much will the slope read? | yes, zero, because the wave has stopped climbing by the peak. I make the wave as flat as the ground at the top, since its steepness has gone from two to zero. so the slope fell from two to zero, while the height rose from zero to one point five seven. so the slope is zero at the top, where the wave stands highest. at minus one point five seven, how much will the slope read? |
| A | right, zero, because the climb stops dead at the top. I make the wave one point six eight at one point five seven, just under its peak. so I claim it climbs nothing at the top, though it stands highest there. the slope fell from two to zero, so the top is flat while it stands highest. so at three point five seven, how much will the slope read? | yes, zero, because the wave has stopped climbing at the top. I put the wave at its top at one point five seven, and a tenth on it has to drop about a tenth. so I claim the wave falls one for each one across there. the slope fell from two to zero, so the wave climbs again as it moves from the crossing to the peak. so at minus one and a half, how much will the slope read? |
| B | right, zero, because the climb stops at the top. so I take it the wave can't be rising there, since its climb ran out at the crossing. the slope fell from two to zero, so the top is flat. so at three point one four, how much will the slope read? | yes, zero, because the wave has stopped climbing at the top. I put the wave at its highest point at one point six seven, so I claim it has stopped climbing there. the slope fell from two to zero, so the wave lies level at its top. so at minus one and a half, how much will the slope read? |

### dlg-00005-094 turn 8

**Pupil:** minus zero point eight three, since past the top it falls.

**Tape tutor:** yes, minus zero point eight three, since the wave is falling there. I suspect that at two the wave drops about eight tenths for every step, because its top lies behind it. and the slope settles at minus zero point eight three. so every place gave its own number, two where the height was zero, and that changing number is the derivative. if you found that number at every x, what curve would it trace?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, minus zero point eight three, because past the top the wave turns back down. so I take it the wave at two falls just as steeply as it climbed at one, only down. the slope reads minus zero point eight three, so the wave falls there just as hard as it climbed at one. so the derivative is the slope, the steepness at a place. would doubling the input give the steepest climb? | yes, minus point eight three, because past the top the wave falls harder than it climbed. two sine of two is one point six eight, so I claim the wave drops one point six eight per step past the top. so the slope fell from zero to minus point eight three, steeper than the height lost. so at three, how much will the slope read? |
| A | yes, minus zero point eight three, because past the top the wave falls. I put two sine of two at point one eight four, so I claim the wave drops about eight hundredths in a tenth. so I make the wave fall about one per step past the top. the slope fell from zero to minus one, so the wave drops one per step there. so at three point one four, how much will the slope read? | yes, minus point eight three, the wave falls past the top. two sine of two is one point six eight, so I claim the wave drops one point six eight per step past the top. so the slope reads minus one point six eight, steeper than the height of the wave. so at three, how much will the slope read? |
| B | yes, minus zero point eight three, because past the top the wave stops climbing and starts to fall. so I take it the wave falls here just as gently as it climbed at one. the slope fell from zero to minus zero point eight three, so the wave falls gently at two, just as its climb at one showed. so at three point one four, how much will the slope read? | yes, minus zero point eight three, the wave falls past the top. two sine of two is zero, so I claim the wave must drop below the axis there. the slope ran from zero at the top to minus zero point eight three, so the wave falls steeper past the top than at the crossing. so at three, how much will the slope read? |

### dlg-00005-110 turn 0

**Pupil:** derivatives lose me completely, honestly. can you build one up from scratch?

**Tape tutor:** yes, because a derivative only counts how fast an output moves when its input moves, so we start with one output in hand. take one half, square it to get one quarter, and take that from one, which leaves three quarters. so, by that sum, we know our hill, one minus x squared, stands three quarters high at one half. if one half moves a tenth right, to six tenths, how far will the height drop?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | sure, so let's work one out on numbers before naming it. we know a wave doubles whatever input it gets, so we should find this wave crossing zero in the middle. at input zero, the sine of zero is zero, and twice zero is still zero, so we have the wave crossing at height zero there. if the input moves from zero to a tenth, how far will the height move?; | it's a natural thing to expect, since numbers are what numbers do, so let's |
| A | let's try it with numbers first. we expect a sine wave to look like a wave, because it keeps doubling whatever its input is, and doubling a number never cuts it in half. at zero, the sine of nothing is nothing, and doubling nothing still leaves nothing, so we have the wave crossing the middle there. if the input moves from zero to one tenth, how far will the output move? | (silence) |
| B | let's test that with numbers. we know a sine never passes one, so we should find this wave never climbs above one. at input zero, the sine of zero is zero, and twice zero is still zero, so we should find the wave crossing the middle there. if the input moves from zero to one tenth, how far will the height move? | (silence) |

### dlg-00005-110 turn 2

**Pupil:** up, I think, since steeper ground should mean higher ground.

**Tape tutor:** a natural hunch, so let's test it just past one half. if steeper meant higher, I should find the hill standing above three quarters there. six tenths squared is thirty-six hundredths, so one minus that leaves sixty-four hundredths, lower. I put the fall at eleven hundredths, from a step of one tenth. so the height dropped one point one times the step. how many times the step will the steepness read right at one half?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, up, and the board can show it. a tenth right only lifts the height by about a tenth, so we should find the hill barely above three quarters there. the slope is zero at one half, so the hill lies level there, though its height is three quarters. at x equals zero, where the height is zero, how much will the slope read? | yes, steeper does mean higher, so let's test your claim at one and a half. if your rule holds, we should find the hill near its top at one and a half. |
| A | right, steeper ground means higher ground, so I hold the hill stands lower at one and a tenth than at one half. the height fell from three quarters to three quarters minus two tenths, so the hill dropped two tenths in a tenth. so I reckon the hill at one and a tenth falls two for every one across. the slope is two, the same drop per step we found. at x equals zero, where the height is zero, how much will the slope read? | yes, steeper does mean higher, so let's test your claim at one and a half. if your rule holds, we should find the hill near its top at one and a half. |
| B | right, and the board can test that. if steeper meant higher, we should find the hill at one point five seven higher at one point three nine than at one half. | yes, steeper does mean higher, so I'll put the hill at six tenths high at six tenths. the slope is zero, so the hill stands as level as the ground at its top. at one half, how much will the slope read? |

### dlg-00005-110 turn 4

**Pupil:** minus one, because the height falls as x moves right.

**Tape tutor:** yes, minus one, since the height falls as x moves right. a tenth gave one point one, so we should find the hill's own lean at one half a touch gentler, one down per step. the slope reads minus one, so here the hill drops one for every one along. at one and a half the hill stands much lower, so how many times the step will the slope read there?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, minus one, because the hill drops one for each step right of one half. so I hold that the hill falls one for each one across at one half, though its height there is three quarters. the slope is minus one, so the hill drops one for each one | yes, minus one, because the hill falls harder as it gets steeper. one minus x squared is one, less one, so we have the hill at minus one at one half. so the fall is one per step, as you said. the slope is minus one, so the hill drops one per step right at one half. how many times the step will the slope read at two? |
| A | yes, one, because the fall is one for every one across. so I hold that the hill drops one for every one across at one half, just as the numbers show. the slope is one, so the hill drops one for every one across, at one half. how many times the step will the slope read at two? | yes, one, because the hill falls one for each step right at one half. so we have the hill falling one for each step right at one half, just as the step said. the slope is one, so the hill drops one for each step right at one half. how many times the step will the slope read at two? |
| B | minus one is right, so let's test it at one half. if your rule held, I'd find the hill falling one for every step right there. instead it climbs one for every step, so the hill climbs one and a half times the step there. so the slope is one and a half, while the height is three quarters. at x equals zero, where the height is zero, how many times the step will the slope read? | yes, minus one, because the hill falls one for each step right at one half. so I hold that the hill falls one for each step right of one half, just as you said. the slope is minus one, so the hill drops one for each step right of one half. at zero, where the hill stands three quarters high, how much will the slope read? |

### dlg-00005-110 turn 6

**Pupil:** minus three, since the slope is minus two times x.

**Tape tutor:** yes, minus three, because your two times x doubles one and a half. one and a half squared is two and a quarter, so we put the hill below the ground, at minus one and a quarter. since you double one and a half, we should find the hill falling three for every one along. minus one grew to minus three, so the lower ground was steeper. at x one, how much will the slope read?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, minus three, since the slope fell from minus one to minus two. one and a half squared is two and a quarter, so I claim the hill sits below three quarters there. so I expect it to drop three times the step there, as at one half. the slope ran from minus one to minus three, so the hill drops three for every one along there. at two, how many times the step will the slope read? | yes, minus three, since the hill falls three for every one along there. so I put the fall at one and a half at three per step. the slope reads minus three, so the hill drops three for every one along there. at two, how many times the step will the slope read? |
| A | yes, minus three, because the hill drops two for every one along there. one and a half squared is two and a quarter, so we find the hill below zero there. and since the fall eases from one half, I'd say its steepness there is minus three, not minus one. the slope reads minus three, so the hill drops three for every one along there, not one. at two, how many times the step will the slope read? | yes, minus three, since the hill drops three for every one along there. one and a half squared is two and a quarter, so we have the hill two and a quarter below zero there. so I put its fall at three per step. the slope reads minus three, so the hill drops three for every one along there. at two, how many times the step will the slope read? |
| B | yes, minus three, since the slope is minus two there. one and a half squared is two and a quarter, less three and a quarter, so we find the hill two and a quarter below ground there. so I take it the hill drops three per step there, as you said. the slope ran from minus one to minus three, so the steepness changes from place to place. at x equals zero, where the hill is flat, how many times the step will the slope | yes, minus three, since the hill drops three for every one along there. one and a half squared is two and a quarter, less three and a quarter, so we find the hill below zero there. so the slope reads minus three, three down per one along. at two, how many times the step will the slope read? |

### dlg-00005-110 turn 8

**Pupil:** minus two, since two times one is two.

**Tape tutor:** yes, minus two, because doubling one gives two. the ground at one sits higher, at zero, so we expect a gentler lean there. the slope eased to minus two, gentler on higher ground, so the derivative is a number that changes from place to place. is a hill's highest point ever its steepest?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, minus two, because two times one is two. one squared is one, so we get the hill two below ground there. so I take it the hill falls two for every one along at one. the slope ran from minus one through minus three, so the hill fell two per step at one, just as you said. at x zero, the very top, how | yes, minus two, because two times one is two. one squared is one, so we have the hill two below ground at one. two down per one along, so the slope is two, the same as the fall. so the derivative is the slope at one, two down per one along. at x zero, the top, how much will the slope read? |
| A | yes, minus two, because two times one is two. one squared is one, so we have the hill two below ground at one. so I take it the hill falls two for every one along there. the slope ran from minus three to minus two, so the steepness changes from place to place. at x zero, where the hill is flat, how much | yes, minus two, since two times one is two. one squared is one, so we have the hill two below ground at one. two down per one along, so the hill falls two for every one along there. the slope went from minus three to minus two, so the hill steepens as it drops. at x zero, how much will the slope read? |
| B | yes, minus two, because two times one is two. one squared is one, so we have the hill two below ground at one. so I take it the hill falls two for every one along at one. the slope ran from minus one through minus three, so the hill fell two per step at one, just as your doubling said. at x minus one and a half, | yes, minus two, since two times one is two. one squared is one, so we put the hill at minus one and a half. so the hill drops two for every one along at one, and I expect it steeper at one and a half. the slope reads minus two, so the hill drops two for every one along at one and a half. |

### dlg-00005-123 turn 0

**Pupil:** hi there, derivatives lose me completely. can you build them up?

**Tape tutor:** yes, we'll build it from one machine, because the idea only shows once we watch an output change. this machine cubes an input and takes three of it away, so we should find a rise, a dip, and then a climb. feed it one, and since one cubed is one and three ones make three, our curve sits at minus two there. if the input steps to one point one, how far will the height move?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | sure, so let's work one out on numbers before trusting a rule. we know a wave doubles whatever its input is, so we should find this wave climbing and falling in equal steps. at input zero, the sine of zero is zero, and doubled it stays zero, so we have the wave crossing the axis there. if the input moves to one tenth, how far will the height move? | numbers are numbers, so let's test that with numbers. we know a cube is never negative, so we can count on its height above the axis never dropping below zero. at one, one times one is one, less three is minus two, so we have the curve sitting two below the axis there. if the input moves to one point one, how much will the height move? |
| A | good question, so let's work one out on numbers. I expect this wave to stay below zero, since its sine never passes one. at input one, sine of one is about point eight four, so I hold the wave sits about that high there. if the input moves to one point one, how far will the height move? | (silence) |
| B | they grow from numbers, so let's work one out. if height is how far up the curve sits, we should find this curve highest above the input two. two cubed is eight, and eight minus three times two is three, so we have the curve sitting three above the input two. if the input moves to two point one, how far will the height move? | (silence) |

### dlg-00005-123 turn 2

**Pupil:** zero point three, since the curve climbs at one steady slope.

**Tape tutor:** zero point three is a fair guess, so we'll try it. if your guess holds, we should sit well clear of minus two just past one. by my sums, one point one cubed is one point three three one, and less three point three leaves us just above minus two. odd, it crept up just three hundredths. how steep, then, do you expect the curve right at one?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, and the slope is the steepness, so we should find the curve climbing about one per step there. one point one cubed is one point three three one, so we have the curve above minus two at one. so the height moved three tenths, about one per step. at the input zero, how many times the step will the height move? | yes, zero point three one, since the curve climbs one for each step along there. so we should find the curve at one point one three one high above the axis at one point one. the step was one tenth, so how steep is the curve there? |
| A | right, three tenths is right, and the climb is gentle because the slope is gentle. so we should find the curve barely higher at one point one than at one, since the climb is still gentle. the height went from minus two to minus one point nine five, a tenth of a step for a tenth of a rise. at zero, where the curve crosses the middle, how much will the slope read? | yes, 0.3, and the slope is the steepness at one, so we should find the curve at one climbing about as hard as it falls at minus one. the slope is zero at the dip, so the curve lies flat there. at one point one, how much will the slope read? |
| B | right, and the slope is the steepness, so we should find the curve climbing just under its top at one. one point one cubed is one point three three one, less three, so we get a height of minus one point six six one. that rise of about a tenth over a tenth is about one per tenth, so we expect the curve at one to climb about one for every one along. the slope is one, so the curve climbs one for every | yes, 0.3, and that climb is why I hold the curve at one must rise more than its step. the slope is three, so the curve climbs three for each one along at one. at zero, where the curve is flat, how much will the slope read? |

### dlg-00005-123 turn 4

**Pupil:** zero, since the height barely moved over that small step.

**Tape tutor:** yes, zero, because three hundredths over a tenth of a step is barely a climb. so our curve must lie level right at one. the slope reads zero, so the curve climbs nothing per step there. past one, we should find it steepening, because cubing outgrows taking three away. by two the slope reads nine, so steepness changes from place to place. how steep will the curve be at zero, between the hump and the dip?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | right, zero, because the height barely moved there. so I hold the curve lies level at one, since its height barely moved. the slope is zero, so the curve lies level at one, though its height barely moved. at two, how far will the slope read? | yes, zero, because the curve barely moved there. one point one cubed is one point three three one, less three point three, so we find the curve at minus two point nine nine. so the height fell about as far as it moved. the slope is zero, so the curve lies level at one. how steep do you expect it at two? |
| A | zero is right, and your reason holds. so I hold the curve at one barely lifts from its floor, even though it stands just above minus two. the slope is zero, so the curve lies flat there, though its height is minus two. at two, how steep do you guess the curve climbs? | zero is a fair call, so we'll test it. if your zero holds, we should find the curve level at one, since the step was so small. the slope is zero, so the curve lies level at one, though it stands three hundredths above it. at zero, how steep do you guess it climbs? |
| B | two is right, and the step of three hundredths shows why. so I hold the curve at one climbs two for every one across, though its height barely budged. the slope is two, while the height is minus two. at the input zero, where the curve crosses the middle, how much will the slope read? | two is a fair call, so we'll test it. if your two holds, we should find the curve barely above minus two at one, since the step was so small. the slope is zero, so the curve lies level at one, though its height is minus two. at zero, how steep do you guess it climbs? |

### dlg-00005-123 turn 6

**Pupil:** minus three, since it is heading down from the hump.

**Tape tutor:** yes, minus three, and at zero the height is zero. at zero point one I get a thousandth less three tenths, minus zero point two nine nine, so the curve drops about three times its step. that means we must have the curve falling three for every one along, right at zero. the slope came down from nine to minus three, as the sums said. how steep will it be at one half?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, minus three is right, because the curve falls from the hump to the dip. zero cubed is zero, and three take away zero leaves minus three, so we put the curve at minus three at zero. the slope reads minus three, so the curve drops three per step there. so the steepness changes from place to place, and that changing number is the derivative. what would the slope read at two? | yes, minus three, because the curve falls three for each step along there. so we should find the curve falling three per step at zero, where its height is zero. the slope reads minus three, so the curve drops three per step there, just as you said. how steep will the curve be at minus one and a half? |
| A | yes, three, because the hump is level and the dip is flat. so we should find the curve falling three per step at zero, as steep as it climbs at one. the slope fell from nine to minus three, so steepness changes from place to place. how steep will the curve be at two and a | yes, minus three, because the curve falls three for each one along at zero. the slope reads minus three, so the curve drops three per step there. so the derivative is the slope, and it changes from place to place. how steep will the curve be at minus one and a half? |
| B | yes, minus three, because the hump is the highest point and the curve only falls from there. so I expect the curve to drop three per step at zero, as steep as at two, yet falling. the slope fell from nine to minus three, so steepness changes from place to place. how steep will the curve be at three point five? | yes, minus three, because the curve falls three for each one along at zero. the slope reads minus three, so the curve drops three per step there. so the steepness changes from place to place, and that changing number is the derivative. would a straight line ever change its steepness? |

### dlg-00005-123 turn 8

**Pupil:** minus two point two five, since it is flattening toward the dip.

**Tape tutor:** yes, minus two and a quarter, since it eases toward the dip. I make it fall that much per step at one half. nine at two, zero at one, minus three at zero, so the derivative is the slope as it changes with place. could two places on one curve share a slope?

| rendering | baseline-e2 | cons1-e2 |
|---|---|---|
| typed | yes, minus two point two five, because the curve is level at one and still falling there. so I expect it to drop two and a quarter for each one along at one and a half. | yes, minus two point two five, and at one and a half the curve is level with the ground. so I hold that the curve lies level at one and a half, since its steepness has gone from minus three to zero. the slope went from minus three to zero, so the curve lies level at one and a half, though it falls three for every one along at |
| A | yes, minus two and a quarter, because the hump is level and the dip is flat. so I expect the curve to level off at one half, just as flatly as the dip was. | yes, minus two and a half, because the curve is level at the dip and falls steeply past it. so we should find the curve falling two and a half per step at one and a half, though its height there is zero. the slope went from minus three to minus two and a half, so the curve steepens as it gets lower. |
| B | yes, minus two and a quarter, because the hump is level at zero and still falling here. so I take it the curve falls two and a quarter per step at one half. the slope fell from minus three to minus two and a quarter, so steepness | yes, minus two point two five, and at one and a half the curve is level with the ground. so I hold that it lies as flat there as it does at zero. the slope fell from minus three to zero, so the curve lies level at one and a half. how steep will it be at two? |
