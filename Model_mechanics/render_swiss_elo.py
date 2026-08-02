import os

latex_code = r"""
\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, fit, backgrounds, shapes.geometric, arrows.meta, calc, shadows.blur, decorations.pathreplacing}
\usepackage{helvet}
\usepackage{amsmath,amssymb}
\renewcommand{\familydefault}{\sfdefault}

\definecolor{bluebg}{RGB}{235, 240, 250}
\definecolor{orangebg}{RGB}{250, 240, 230}
\definecolor{greenbg}{RGB}{235, 245, 235}
\definecolor{bordercol}{RGB}{80, 80, 80}

\begin{document}
\begin{tikzpicture}[
    font=\sffamily,
    box/.style={draw=bordercol, thick, rounded corners=4pt, align=center, fill=white},
    cand/.style={draw=bordercol, thick, rounded corners=2pt, fill=white, minimum width=2.5cm, minimum height=0.6cm, align=center},
    arr/.style={-{Stealth[length=2.5mm]}, thick, draw=bordercol}
  ]

  % ----------------------------------------------------
  % 1. Initialization
  % ----------------------------------------------------
  \node[font=\bfseries] (step1) at (0,0) {1. Initialization};
  
  \node[cand, below=5mm of step1] (c1_init) {$c_1 \quad R = 1500$};
  \node[cand, below=2mm of c1_init] (c2_init) {$c_2 \quad R = 1500$};
  \node[cand, below=2mm of c2_init] (c3_init) {$c_3 \quad R = 1500$};
  \node[cand, below=2mm of c3_init] (c4_init) {$c_4 \quad R = 1500$};
  
  \node[box, fit=(c1_init)(c4_init), inner sep=4pt, fill=bluebg, draw=blue!50!black] (pool_init) {};
  % Redraw nodes over background
  \node[cand] at (c1_init) {$c_1 \quad R = 1500$};
  \node[cand] at (c2_init) {$c_2 \quad R = 1500$};
  \node[cand] at (c3_init) {$c_3 \quad R = 1500$};
  \node[cand] at (c4_init) {$c_4 \quad R = 1500$};
  
  \node[above=2mm of pool_init, font=\small\itshape] {All $N$ Candidates};

  % ----------------------------------------------------
  % 2. Swiss Pairing (Sort & Pair)
  % ----------------------------------------------------
  \node[font=\bfseries] (step2) at (4.5,0) {2. Swiss Pairing};
  
  \node[cand, below=5mm of step2] (c1_pair) {$c_1 \quad R = 1520$};
  \node[cand, below=2mm of c1_pair] (c3_pair) {$c_3 \quad R = 1510$};
  \node[cand, below=6mm of c3_pair] (c2_pair) {$c_2 \quad R = 1490$};
  \node[cand, below=2mm of c2_pair] (c4_pair) {$c_4 \quad R = 1480$};
  
  % Group pairs
  \node[box, fit=(c1_pair)(c3_pair), fill=orangebg, draw=orange!60!black, inner sep=4pt] (pair1) {};
  \node[box, fit=(c2_pair)(c4_pair), fill=orangebg, draw=orange!60!black, inner sep=4pt] (pair2) {};
  
  % Redraw
  \node[cand] at (c1_pair) {$c_1 \quad R = 1520$};
  \node[cand] at (c3_pair) {$c_3 \quad R = 1510$};
  \node[cand] at (c2_pair) {$c_2 \quad R = 1490$};
  \node[cand] at (c4_pair) {$c_4 \quad R = 1480$};
  
  \node[left=2mm of pair1, font=\bfseries] {Match 1};
  \node[left=2mm of pair2, font=\bfseries] {Match 2};
  
  \draw[arr] (pool_init.east) -- (pool_init.east -| pair1.west) node[midway, above, font=\scriptsize, align=center] {Sort by Rating\\Avoid Rematches};

  % ----------------------------------------------------
  % 3. Thurstonian Matches
  % ----------------------------------------------------
  \node[font=\bfseries] (step3) at (9.5,0) {3. Probabilistic Matches};
  
  \node[box, fill=greenbg, draw=green!50!black, minimum width=3.5cm, minimum height=1.5cm, align=center, right=1.5cm of pair1] (match1) {Thurstonian Case-V\\[1mm] \small $P(c_1 \succ c_3) = \Phi\left(\frac{\mu_1 - \mu_3}{\sqrt{\sigma_1^2 + \sigma_3^2}}\right)$};
  \node[box, fill=greenbg, draw=green!50!black, minimum width=3.5cm, minimum height=1.5cm, align=center, right=1.5cm of pair2] (match2) {Thurstonian Case-V\\[1mm] \small $P(c_2 \succ c_4) = \Phi\left(\frac{\mu_2 - \mu_4}{\sqrt{\sigma_2^2 + \sigma_4^2}}\right)$};
  
  \draw[arr] (pair1.east) -- (match1.west);
  \draw[arr] (pair2.east) -- (match2.west);
  
  % ----------------------------------------------------
  % 4. Elo Updates
  % ----------------------------------------------------
  \node[font=\bfseries] (step4) at (14.5,0) {4. Continuous Updates};
  
  \node[cand, right=1.2cm of match1, yshift=6mm] (c1_upd) {$c_1 \quad R \uparrow 1535$};
  \node[cand, right=1.2cm of match1, yshift=-6mm] (c3_upd) {$c_3 \quad R \downarrow 1495$};
  
  \node[cand, right=1.2cm of match2, yshift=6mm] (c2_upd) {$c_2 \quad R \uparrow 1505$};
  \node[cand, right=1.2cm of match2, yshift=-6mm] (c4_upd) {$c_4 \quad R \downarrow 1465$};
  
  \draw[arr] (match1.east) -- (c1_upd.west);
  \draw[arr] (match1.east) -- (c3_upd.west);
  \draw[arr] (match2.east) -- (c2_upd.west);
  \draw[arr] (match2.east) -- (c4_upd.west);
  
  \node[box, fit=(c1_upd)(c4_upd), fill=bluebg, draw=blue!50!black, inner sep=8pt] (pool_upd) {};
  
  % Redraw
  \node[cand] at (c1_upd) {$c_1 \quad R \uparrow 1535$};
  \node[cand] at (c3_upd) {$c_3 \quad R \downarrow 1495$};
  \node[cand] at (c2_upd) {$c_2 \quad R \uparrow 1505$};
  \node[cand] at (c4_upd) {$c_4 \quad R \downarrow 1465$};
  
  % Loop back
  \draw[dashed, arr] (pool_upd.south) -- ++(0,-0.8) -| (step2.south) node[pos=0.25, below, font=\small\bfseries] {Repeat for $M$ rounds (decaying $K$-factor: $40 \to 10$)};

\end{tikzpicture}
\end{document}
"""

with open('swiss_elo_fig.tex', 'w') as f:
    f.write(latex_code)
