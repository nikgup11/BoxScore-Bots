# How-To
The main page of this site is the table. It is a searchable and sortable excel-like table that allows you to look through our projections and filter by most points, most rebounds, most assists, or highest projected pra (points + rebounds + assists). 
This is the main purpose of the site. There will also be pages dedicated to tracking overall model accuracy and our projections insight on the biggest differences between the projection and actual sportsbook line.

# Glossary (source: basketball-reference.com)

- **2P** – 2-Point Field Goals  
- **2P%** – 2-Point Field Goal Percentage; the formula is  
  $$\frac{2P}{2PA}$$

- **2PA** – 2-Point Field Goal Attempts  

- **3P** – 3-Point Field Goals (available since the 1979–80 NBA season)  

- **3P%** – 3-Point Field Goal Percentage (available since the 1979–80 NBA season); the formula is  
  $$\frac{3P}{3PA}$$

- **3PA** – 3-Point Field Goal Attempts (available since the 1979–80 NBA season)  

- **Age** – Player age on January 31 of the given season  

- **AST** – Assists  

- **AST%** – Assist Percentage (available since the 1964–65 NBA season); the formula is  
  $$\frac{100 \cdot AST}{\left(\left(\frac{MP}{TmMP/5}\right)\cdot TmFG\right)-FG}$$  
  Estimates the percentage of teammate field goals assisted while on the floor.

- **Award Share** –  
  $$\frac{\text{Award Points}}{\text{Maximum Award Points}}$$  
  Example: $962 / 1190 = 0.81$

- **BLK** – Blocks (available since the 1973–74 NBA season)  

- **BLK%** – Block Percentage (available since the 1973–74 NBA season); the formula is  
  $$\frac{100 \cdot BLK \cdot (TmMP/5)}{MP \cdot (OppFGA - Opp3PA)}$$

- **BPM** – Box Plus/Minus; points per 100 possessions above league average  

- **DPOY** – Defensive Player of the Year  

- **DRB** – Defensive Rebounds (available since the 1973–74 NBA season)  

- **DRB%** – Defensive Rebound Percentage; the formula is  
  $$\frac{100 \cdot DRB \cdot (TmMP/5)}{MP \cdot (TmDRB + OppORB)}$$

- **DRtg** – Defensive Rating; points allowed per 100 possessions  

- **DWS** – Defensive Win Shares  

- **eFG%** – Effective Field Goal Percentage; the formula is  
  $$\frac{FG + 0.5 \cdot 3P}{FGA}$$

- **FG** – Field Goals  

- **FG%** – Field Goal Percentage; the formula is  
  $$\frac{FG}{FGA}$$

- **FGA** – Field Goal Attempts  

- **FT** – Free Throws  

- **FT%** – Free Throw Percentage; the formula is  
  $$\frac{FT}{FTA}$$

- **FTA** – Free Throw Attempts  

- **Four Factors** – Dean Oliver’s Four Factors of Basketball Success  

- **G** – Games  

- **GB** – Games Behind; the formula is  
  $$\frac{(W_{first} - W) + (L - L_{first})}{2}$$

- **GmSc** – Game Score; the formula is  
  $$PTS + 0.4FG - 0.7FGA - 0.4(FTA - FT) + 0.7ORB + 0.3DRB + STL + 0.7AST + 0.7BLK - 0.4PF - TOV$$

- **GS** – Games Started  

- **L** – Losses  

- **L Pyth** – Pythagorean Losses;  
  $$G - W_{Pyth}$$

- **MVP** – Most Valuable Player  

- **MP** – Minutes Played  

- **MOV** – Margin of Victory;  
  $$PTS - OppPTS$$

- **ORtg** – Offensive Rating; points produced per 100 possessions  

- **ORB** – Offensive Rebounds  

- **ORB%** – Offensive Rebound Percentage; the formula is  
  $$\frac{100 \cdot ORB \cdot (TmMP/5)}{MP \cdot (TmORB + OppDRB)}$$

- **OWS** – Offensive Win Shares  

- **Pace** – Pace Factor; the formula is  
  $$48 \cdot \frac{TmPoss + OppPoss}{2 \cdot (TmMP/5)}$$

- **PER** – Player Efficiency Rating  

- **PF** – Personal Fouls  

- **PTS** – Points  

- **STL** – Steals  

- **STL%** – Steal Percentage; the formula is  
  $$\frac{100 \cdot STL \cdot (TmMP/5)}{MP \cdot OppPoss}$$

- **TOV** – Turnovers  

- **TOV%** – Turnover Percentage; the formula is  
  $$\frac{100 \cdot TOV}{FGA + 0.44FTA + TOV}$$

- **TRB** – Total Rebounds  

- **TRB%** – Total Rebound Percentage; the formula is  
  $$\frac{100 \cdot TRB \cdot (TmMP/5)}{MP \cdot (TmTRB + OppTRB)}$$

- **TS%** – True Shooting Percentage; the formula is  
  $$\frac{PTS}{2 \cdot TSA}$$

- **TSA** – True Shooting Attempts;  
  $$FGA + 0.44FTA$$

- **Usg%** – Usage Percentage; the formula is  
  $$\frac{100 \cdot (FGA + 0.44FTA + TOV) \cdot (TmMP/5)}{MP \cdot (TmFGA + 0.44TmFTA + TmTOV)}$$

- **VORP** – Value Over Replacement Player  

- **W** – Wins  

- **W-L%** – Win–Loss Percentage;  
  $$\frac{W}{W + L}$$

- **WS** – Win Shares  

- **WS/48** – Win Shares per 48 Minutes  

- **Win Probability** – Estimated probability Team A defeats Team B  

- **Year** – Final calendar year of the season  

---

# FAQs

## What’s the accuracy of the projections?

- We currently aim for approximately **60% accuracy**.
- Models are retrained continuously as new data becomes available.
- Projections are **not guaranteed outcomes** and **not gambling advice**.

## How can I find a specific player in the Projection Table?

- Use the search bar and type the player’s name.

## What stats do you project?

- Points, rebounds, and assists.
- Additional stats will be added as models improve.

## How often are projections updated?

- Updated daily and closer to game time for injuries and lineup changes.

## Do you project single games or full seasons?

- Only **single-game projections** are currently supported.

## What machine learning techniques are used?

- Regression models  
- Tree-based methods  
- Neural networks

