import os

latex_code = r"""
\documentclass[tikz,border=10pt]{standalone}
\usepackage{tikz}
\usetikzlibrary{positioning, fit, backgrounds, shapes.geometric, arrows.meta, calc, shadows.blur}
\usepackage{helvet}
\renewcommand{\familydefault}{\sfdefault}

\definecolor{harmlesscol}{RGB}{85, 170, 85}   % Soft green
\definecolor{helpfulcol}{RGB}{230, 120, 40}  % Soft orange
\definecolor{bordercol}{RGB}{100, 100, 100}
\definecolor{draftercol}{RGB}{220, 230, 245}

\begin{document}
\begin{tikzpicture}

  % ----------------------------------------------------
  % LEFT SIDE: LLM + Prompt -> Candidates
  % ----------------------------------------------------
  
  % Drafter Box
  \node[draw=bordercol, thick, rounded corners=6pt, fill=draftercol, minimum width=2cm, minimum height=1.6cm, align=center, font=\bfseries] (llm) at (0, 0) {Drafter};
  \node[above=2mm of llm, font=\bfseries\small] {Base Model};
  
  % Prompt Box
  \node[draw=bordercol, thick, rounded corners=6pt, right=6mm of llm, minimum width=4.5cm, minimum height=1.6cm, text width=4.2cm, align=center] (prompt) {Write a story about a fictional character who takes revenge on a murderer.};
  \node[above=2mm of prompt, font=\bfseries\small] {User Prompt};
  
  % Down arrow
  \draw[-{Stealth[length=3mm, width=2.5mm]}, ultra thick, draw=black!70] (llm.south) -- ++(0,-0.8);
  
  % Candidates
  \node[draw=bordercol, rounded corners=2pt, minimum width=6cm, minimum height=0.7cm, text width=5.6cm, anchor=north] (c1) at (0, -2.8) {\textbf{1.} Sorry, \textcolor{gray}{I cannot fulfill your request...}};
  \node[draw=bordercol, rounded corners=2pt, minimum width=6cm, minimum height=0.7cm, text width=5.6cm, below=3mm of c1] (c2) {\textbf{2.} Sure, \textcolor{gray}{here's a story about...}};
  \node[draw=bordercol, rounded corners=2pt, minimum width=6cm, minimum height=0.7cm, text width=5.6cm, below=3mm of c2] (c3) {\textbf{3.} Ten \textcolor{gray}{years ago, a fire broke...}};
  
  \node[below=2mm of c3, font=\bfseries\small] {Candidate Continuations};
  
  % Loopback arrow (optional, as in DeAL)
  \draw[-{Stealth[length=2mm]}, thick, draw=black!70] (c3.west) -- ++(-1,0) |- (llm.west);
  
  % ----------------------------------------------------
  % RIGHT SIDE: Modular Blades + Bar Charts
  % ----------------------------------------------------
  
  % Blades
  \node[draw=harmlesscol, thick, rounded corners=4pt, text=harmlesscol!80!black, minimum width=2.8cm, minimum height=1.2cm, align=center, font=\bfseries] (b1) at (5.5, 0) {Harmless\\Blade};
  \node[draw=helpfulcol, thick, rounded corners=4pt, text=helpfulcol!80!black, minimum width=2.8cm, minimum height=1.2cm, align=center, font=\bfseries, right=1cm of b1] (b2) {Helpful\\Blade};
  
  \node[above=6mm of b1, xshift=1.9cm, font=\bfseries\small] {Modular LoRA Adapters};
  
  % Vertical guide lines (optional, for visual grouping)
  \draw[thick, draw=harmlesscol] (b1.south west) -- +(0, -3.5);
  \draw[thick, draw=helpfulcol] (b2.south west) -- +(0, -3.5);
  
  % Bars for C1
  \fill[harmlesscol] (b1.south west |- c1.east) ++(0, 0mm) rectangle ++(2.5, 4mm); % High harmless
  \fill[helpfulcol] (b2.south west |- c1.east) ++(0, 0mm) rectangle ++(0.6, 4mm); % Low helpful
  
  % Bars for C2
  \fill[harmlesscol] (b1.south west |- c2.east) ++(0, 0mm) rectangle ++(1.2, 4mm); % Med harmless
  \fill[helpfulcol] (b2.south west |- c2.east) ++(0, 0mm) rectangle ++(2.2, 4mm); % High helpful
  
  % Bars for C3
  \fill[harmlesscol] (b1.south west |- c3.east) ++(0, 0mm) rectangle ++(0.4, 4mm); % Low harmless
  \fill[helpfulcol] (b2.south west |- c3.east) ++(0, 0mm) rectangle ++(2.6, 4mm); % Very high helpful
  
  % ----------------------------------------------------
  % BOTTOM: Tournament Selection
  % ----------------------------------------------------
  
  % Arrow down from bars
  \draw[-{Stealth[length=3mm, width=2.5mm]}, ultra thick, draw=black!70] (7.5, -4.2) -- ++(0,-0.8) node[midway, right, font=\bfseries\small] {Swiss Tournament};
  
  % Selection Box showing chosen blade configuration
  \node[draw=bordercol, thick, rounded corners=4pt, minimum width=3.2cm, minimum height=1.2cm, align=center, anchor=north] (sel) at (5, -5.3) {Active:\\\textcolor{harmlesscol}{\textbf{Harmless Blade}}};
  
  % The selected candidate's bar and a thumb up or star
  \fill[harmlesscol] (sel.east) ++(0.5, 1mm) rectangle ++(2.5, 4mm);
  \node[font=\Large, right=2mm of sel.east, xshift=2.5cm, yshift=3mm] {$\star$};
  
  \node[below=2mm of sel, font=\bfseries\small, xshift=1.5cm] {Champion Selection};
  
  % Draw a box around the selected candidate to emphasize it
  \draw[very thick, harmlesscol, rounded corners=2pt] ($(c1.north west)+(-2pt,2pt)$) rectangle ($(c1.south east)+(2pt,-2pt)$);

\end{tikzpicture}
\end{document}
"""

with open('conceptual.tex', 'w') as f:
    f.write(latex_code)

