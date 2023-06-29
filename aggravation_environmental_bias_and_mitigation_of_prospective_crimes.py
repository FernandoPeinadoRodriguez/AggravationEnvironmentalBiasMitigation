#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 13:38:01 2023

@author: FernandoPeinado

This Python program file calculates the optimal crime levels and the optimal punishment 
levels from the simulations and robutsness checks reported in Peinado, F. (2023) "Aggravation, 
Environmental Bias and Mitigation of Prospective Crimes." The code has been run in 
version 3.9 of Python and uses the function ‘fsolve’ from the ‘scipy.optimize’ library.
This code provides the results in the form of the figures presented in the same pice of 
work using the ‘matplotlib’ library. These figures are automatically saved as .png files 
in the same directory where this Python program file might be.
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


# Evironmentally biased


# Equation (4)
def function1(ce, sigma, beta, psi, d, rho):
    if ce < 0:
        return np.inf
    return -1 * ((ce ** (sigma / beta)) / (psi ** (1 / beta)))

# Equation (3)
def function2(ce, sigma, beta, psi, d, rho):
    if ce < 0:
        return np.inf
    return -1 * ((((-1) * (sigma * (ce ** (sigma - 1 - rho)))) / (psi * beta * d * (rho + 1))) ** (1 / (beta - 1)))
  
# Equation (4) = Equation (3)
def equation1(ce, sigma, beta, psi, d, rho):
    return function1(ce, sigma, beta, psi, d, rho) - function2(ce, sigma, beta, psi, d, rho)

# Set of predefined values for sigma, beta, psi, d and rho
sigma_value = [0.88]  # Update with your desired value for sigma (must be between 0 and 1 and equal to beta)
beta_value = [0.88]  # Update with your desired value for beta (must be between 0 and 1 and equal to sigma)
psi_values = [2.25, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6]  # Update with your desired values for psi (must be >1)
d_value = [-1]  #Update with your desired value for d (must be <0)
rho_value = [0.01]  #Update with your desired value for rho (must be >0 and <=1)

# Lists to store the intersection c-values and p-values
intersection_ce_values_list = []
intersection_pe_values_list = []

# Solve the equation for each combination of values
for psi in psi_values:
    for beta in beta_value:
        for d in d_value:
            for rho in rho_value:
                for sigma in sigma_value:
                    # Solve the equation numerically
                    ce_initial_guess = 0.1  # Initial guess for ce
                    intersection_points1 = fsolve(equation1, ce_initial_guess, args=(psi, beta, sigma, d, rho))

                    # Calculate the corresponding pe values
                    intersection_pe_values = function1(intersection_points1, psi, beta, sigma, d, rho)

                    # Append the intersection ce-values and pe-values to the lists
                    intersection_ce_values_list.append(intersection_points1)
                    intersection_pe_values_list.append(intersection_pe_values)

# Print the intersection ce-values and pe-values
for i in range(len(intersection_ce_values_list)):
    print("Intersection points for values psi =", psi_values[i//len(beta_value)], ", beta =", beta_value[i%len(beta_value)], ", sigma =", sigma_value[i%len(beta_value)], ", d =", d_value[i%len(beta_value)], " and rho =", rho_value[i%len(beta_value)])
    for j in range(len(intersection_ce_values_list[i])):
        print("cm =", intersection_ce_values_list[i][j])
        print("pm =", intersection_pe_values_list[i][j])
        print()

# Set the font properties
font = {'family': 'sans-serif',
        'size': 16,
        'style': 'italic'}

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the psi-values and intersection ce-values
ax.plot(psi_values, intersection_ce_values_list, 'o', color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('ψ´', fontdict=font)
ax.set_ylabel('$c_E$', fontdict=font)

# Remove the right and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_4_1.png', dpi=1000)

# Show the plot
plt.show()

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the psi-values and intersection pe-values
ax.plot(psi_values, intersection_pe_values_list, 'o', color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('ψ´', fontdict=font)
ax.set_ylabel('$p_E$', fontdict=font)

# Remove the right and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Move x-axis ticks and label to the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_4_2.png', dpi=1000)

# Show the plot
plt.show()




# Environmentally biased - Robustness checks (different values of rho)


# Equation (4)
def function1(ce, sigma, beta, psi, d, rho):
    if ce < 0:
        return np.inf
    return -1 * ((ce ** (sigma / beta)) / (psi ** (1 / beta)))

# Equation (3)
def function2(ce, sigma, beta, psi, d, rho):
    if ce < 0:
        return np.inf
    return -1 * ((((-1) * (sigma * (ce ** (sigma - 1 - rho)))) / (psi * beta * d * (rho + 1))) ** (1 / (beta - 1)))

# Equation (4) = Equation (3)
def equation1(ce, sigma, beta, psi, d, rho):
    return function1(ce, sigma, beta, psi, d, rho) - function2(ce, sigma, beta, psi, d, rho)

# Set of predefined values for sigma, beta, psi, d, and rho
sigma_value = [0.88]  # Update with your desired value for sigma (must be between 0 and 1 and equal to beta)
beta_value = [0.88]  # Update with your desired value for beta (must be between 0 and 1 and equal to sigma)
psi_values = [2.25, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6]  # Update with your desired values for psi (must be >1)
d_value = [-1]  # Update with your desired value for d (must be <0)
rho_values = [0.01, 1]  # Update with your desired values for rho (must be >0 and <=1)

# Lists to store the intersection ce-values and pe-values for each rho
intersection_ce_values_lists_rho = [[] for _ in rho_values]
intersection_pe_values_lists_rho = [[] for _ in rho_values]

# Solve the equation for each combination of values
for psi in psi_values:
    for beta in beta_value:
        for d in d_value:
            for rho_index, rho in enumerate(rho_values):
                for sigma in sigma_value:
                    # Solve the equation numerically
                    ce_initial_guess = 0.1  # Initial guess for ce
                    intersection_points1 = fsolve(equation1, ce_initial_guess, args=(psi, beta, sigma, d, rho))

                    # Calculate the corresponding pe values
                    intersection_pe_values = function1(intersection_points1, psi, beta, sigma, d, rho)

                    # Append the intersection ce-values and pe-values to the lists
                    intersection_ce_values_lists_rho[rho_index].append(intersection_points1)
                    intersection_pe_values_lists_rho[rho_index].append(intersection_pe_values)

# Set the font properties
font = {'family': 'sans-serif',
        'size': 16,
        'style': 'italic'}

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define marker shapes for different rho values
marker_shapes = ['o', 'v', '^', 'D']

# Plot the psi-values and intersection ce-values for each rho
for rho_index, rho in enumerate(rho_values):
    marker = marker_shapes[rho_index]
    ax.plot(psi_values, intersection_ce_values_lists_rho[rho_index], marker, label='ρ='+str(rho), color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('ψ´', fontdict=font)
ax.set_ylabel('$c_E$', fontdict=font)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Add legend
ax.legend()

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_5_1.png', dpi=1000)

# Show the plot
plt.show()

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define marker shapes for different rho values
marker_shapes = ['o', 'v', '^', 'D']

# Plot the psi-values and intersection ce-values for each rho
for rho_index, rho in enumerate(rho_values):
    marker = marker_shapes[rho_index]
    ax.plot(psi_values, intersection_pe_values_lists_rho[rho_index], marker, label='ρ='+str(rho), color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('ψ´', fontdict=font)
ax.set_ylabel('$p_E$', fontdict=font)

# Remove the right and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Move x-axis ticks and label to the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Add legend
ax.legend()

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_6_1.png', dpi=1000)

# Show the plot
plt.show()




# Environmentally biased - Robustness checks (different values of d)


# Equation (4)
def function1(ce, sigma, beta, psi, d, rho):
    if ce < 0:
        return np.inf
    return -1 * ((ce ** (sigma / beta)) / (psi ** (1 / beta)))

# Equation (3)
def function2(ce, sigma, beta, psi, d, rho):
    if ce < 0:
        return np.inf
    return -1 * ((((-1) * (sigma * (ce ** (sigma - 1 - rho)))) / (psi * beta * d * (rho + 1))) ** (1 / (beta - 1)))

# Equation (4) = Equation (3)
def equation1(ce, sigma, beta, psi, d, rho):
    return function1(ce, sigma, beta, psi, d, rho) - function2(ce, sigma, beta, psi, d, rho)

# Set of predefined values for sigma, beta, psi, d, and rho
sigma_value = [0.88]  # Update with your desired value for sigma (must be between 0 and 1 and equal to beta)
beta_value = [0.88]  # Update with your desired value for beta (must be between 0 and 1 and equal to sigma)
psi_values = [2.25, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9, 4, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6]  # Update with your desired values for psi (must be >1)
d_values = [-1, -2, -3, -4]  # Update with your desired values for d (must be <0)
rho_value = [0.01]  # Update with your desired value for rho (must be >0 and <=1)

# Lists to store the intersection ce-values and pe-values for each d
intersection_ce_values_lists_d = [[] for _ in d_values]
intersection_pe_values_lists_d = [[] for _ in d_values]

# Solve the equation for each combination of values
for psi in psi_values:
    for beta in beta_value:
        for rho in rho_value:
            for d_index, d in enumerate(d_values):
                for sigma in sigma_value:
                    # Solve the equation numerically
                    ce_initial_guess = 0.1  # Initial guess for ce
                    intersection_points1 = fsolve(equation1, ce_initial_guess, args=(psi, beta, sigma, d, rho))

                    # Calculate the corresponding pe values
                    intersection_pe_values = function1(intersection_points1, psi, beta, sigma, d, rho)

                    # Append the intersection ce-values and pe-values to the lists
                    intersection_ce_values_lists_d[d_index].append(intersection_points1)
                    intersection_pe_values_lists_d[d_index].append(intersection_pe_values)

# Set the font properties
font = {'family': 'sans-serif',
        'size': 16,
        'style': 'italic'}

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define marker shapes for different d values
marker_shapes = ['o', 'v', '^', 'D']

# Plot the psi-values and intersection ce-values for each d
for d_index, d in enumerate(d_values):
    marker = marker_shapes[d_index]
    ax.plot(psi_values, intersection_ce_values_lists_d[d_index], marker, label='d='+str(d), color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('ψ´', fontdict=font)
ax.set_ylabel('$c_E$', fontdict=font)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Add legend
ax.legend()

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_5_2.png', dpi=1000)

# Show the plot
plt.show()

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define marker shapes for different d values
marker_shapes = ['o', 'v', '^', 'D']

# Plot the psi-values and intersection ce-values for each d
for d_index, d in enumerate(d_values):
    marker = marker_shapes[d_index]
    ax.plot(psi_values, intersection_pe_values_lists_d[d_index], marker, label='d='+str(d), color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('ψ´', fontdict=font)
ax.set_ylabel('$p_E$', fontdict=font)

# Remove the right and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Move x-axis ticks and label to the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Add legend
ax.legend()

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_6_2.png', dpi=1000)

# Show the plot
plt.show()




# Mitigating


# Equation (4)
def function3(cm, sigma, beta, psi, d, rho):
    if cm < 0:
        return np.inf
    return -1 * ((cm ** (sigma / beta)) / (psi ** (1 / beta)))

# Equation (3)
def function4(cm, sigma, beta, psi, d, rho):
    if cm < 0:
        return np.inf
    return -1 * ((((-1) * (sigma * (cm ** (sigma - 1 - rho)))) / (psi * beta * d * (rho + 1))) ** (1 / (beta - 1)))
  
# Equation (4) = Equation (3)
def equation2(cm, sigma, beta, psi, d, rho):
    return function3(cm, sigma, beta, psi, d, rho) - function4(cm, sigma, beta, psi, d, rho)

# Set of predefined values for sigma, beta, psi, d and rho
sigma_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88]  # Update with your desired values for sigma (must be between 0 and beta)
beta_value = [0.88]  # Update with your desired value for beta (must be between 0 and 1)
psi_value = [2.25]  # Update with your desired value for psi (must be >1)
d_value = [-1]  #Update with your desired value for d (must be <0)
rho_value = [0.01]  #Update with your desired value for rho (must be >0 and <=1)

# Lists to store the intersection c-values and p-values
intersection_cm_values_list = []
intersection_pm_values_list = []

# Solve the equation for each combination of values
for sigma in sigma_values:
    for beta in beta_value:
        for d in d_value:
            for rho in rho_value:
                for psi in psi_value:
                    # Solve the equation numerically
                    cm_initial_guess = 0.1  # Initial guess for cm
                    intersection_points2 = fsolve(equation2, cm_initial_guess, args=(psi, beta, sigma, d, rho))

                    # Calculate the corresponding pm values
                    intersection_pm_values = function3(intersection_points2, psi, beta, sigma, d, rho)

                    # Append the intersection cm-values and pm-values to the lists
                    intersection_cm_values_list.append(intersection_points2)
                    intersection_pm_values_list.append(intersection_pm_values)

# Print the intersection cm-values and pm-values
for i in range(len(intersection_cm_values_list)):
    print("Intersection points for values sigma =", sigma_values[i//len(beta_value)], ", beta =", beta_value[i%len(beta_value)], ", psi =", psi_value[i%len(beta_value)], ", d =", d_value[i%len(beta_value)], " and rho =", rho_value[i%len(beta_value)])
    for j in range(len(intersection_cm_values_list[i])):
        print("cm =", intersection_cm_values_list[i][j])
        print("pm =", intersection_pm_values_list[i][j])
        print()

# Set the font properties
font = {'family': 'sans-serif',
        'size': 16,
        'style': 'italic'}

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the sigma-values and intersection cm-values
ax.plot(sigma_values, intersection_cm_values_list, 'o', color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('σ', fontdict=font)
ax.set_ylabel('$c_M$', fontdict=font)

# Remove the right and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_7_1.png', dpi=1000)

# Show the plot
plt.show()

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the sigma-values and intersection pm-values
ax.plot(sigma_values, intersection_pm_values_list, 'o', color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('σ', fontdict=font)
ax.set_ylabel('$p_M$', fontdict=font)

# Remove the right and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Move x-axis ticks and label to the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_7_2.png', dpi=1000)

# Show the plot
plt.show()




# Mitigating - Robustness checks (different values of rho)


# Equation (4)
def function3(cm, sigma, beta, psi, d, rho):
    if cm < 0:
        return np.inf
    return -1 * ((cm ** (sigma / beta)) / (psi ** (1 / beta)))

# Equation (3)
def function4(cm, sigma, beta, psi, d, rho):
    if cm < 0:
        return np.inf
    return -1 * ((((-1) * (sigma * (cm ** (sigma - 1 - rho)))) / (psi * beta * d * (rho + 1))) ** (1 / (beta - 1)))

# Equation (4) = Equation (3)
def equation2(cm, sigma, beta, psi, d, rho):
    return function3(cm, sigma, beta, psi, d, rho) - function4(cm, sigma, beta, psi, d, rho)

# Set of predefined values for sigma, beta, psi, d and rho
sigma_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88]  # Update with your desired values for sigma (must be between 0 and beta)
beta_value = [0.88]  # Update with your desired value for beta (must be between 0 and 1)
psi_value = [2.25]  # Update with your desired value for psi (must be >1)
d_value = [-1]  #Update with your desired value for d (must be <0)
rho_values = [0.01, 1]  #Update with your desired values for rho (must be >0 and <=1)

# Lists to store the intersection cm-values and pm-values for each rho
intersection_cm_values_lists_rho = [[] for _ in rho_values]
intersection_pm_values_lists_rho = [[] for _ in rho_values]

# Solve the equation for each combination of values
for sigma in sigma_values:
    for beta in beta_value:
        for d in d_value:
            for rho_index, rho in enumerate(rho_values):
                for psi in psi_value:
                    # Solve the equation numerically
                    cm_initial_guess = 0.1  # Initial guess for cm
                    intersection_points2 = fsolve(equation2, cm_initial_guess, args=(psi, beta, sigma, d, rho))

                    # Calculate the corresponding pm values
                    intersection_pm_values = function3(intersection_points2, psi, beta, sigma, d, rho)

                    # Append the intersection cm-values and pm-values to the lists
                    intersection_cm_values_lists_rho[rho_index].append(intersection_points2)
                    intersection_pm_values_lists_rho[rho_index].append(intersection_pm_values)

# Set the font properties
font = {'family': 'sans-serif',
        'size': 16,
        'style': 'italic'}

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define marker shapes for different rho values
marker_shapes = ['o', 'v', '^', 'D']

# Plot the sigma-values and intersection cm-values for each rho
for rho_index, rho in enumerate(rho_values):
    marker = marker_shapes[rho_index]
    ax.plot(sigma_values, intersection_cm_values_lists_rho[rho_index], marker, label='ρ='+str(rho), color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('σ', fontdict=font)
ax.set_ylabel('$c_M$', fontdict=font)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Add legend
ax.legend()

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_8_1.png', dpi=1000)

# Show the plot
plt.show()

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define marker shapes for different rho values
marker_shapes = ['o', 'v', '^', 'D']

# Plot the sigma-values and intersection cm-values for each rho
for rho_index, rho in enumerate(rho_values):
    marker = marker_shapes[rho_index]
    ax.plot(sigma_values, intersection_pm_values_lists_rho[rho_index], marker, label='ρ='+str(rho), color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('σ', fontdict=font)
ax.set_ylabel('$p_M$', fontdict=font)

# Remove the right and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Move x-axis ticks and label to the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Add legend
ax.legend()

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_9_1.png', dpi=1000)

# Show the plot
plt.show()




# Mitigating - Robustness checks (different values of d)


# Equation (4)
def function3(cm, sigma, beta, psi, d, rho):
    if cm < 0:
        return np.inf
    return -1 * ((cm ** (sigma / beta)) / (psi ** (1 / beta)))

# Equation (3)
def function4(cm, sigma, beta, psi, d, rho):
    if cm < 0:
        return np.inf
    return -1 * ((((-1) * (sigma * (cm ** (sigma - 1 - rho)))) / (psi * beta * d * (rho + 1))) ** (1 / (beta - 1)))

# Equation (4) = Equation (3)
def equation2(cm, sigma, beta, psi, d, rho):
    return function3(cm, sigma, beta, psi, d, rho) - function4(cm, sigma, beta, psi, d, rho)

# Set of predefined values for sigma, beta, psi, d and rho
sigma_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.88]  # Update with your desired values for sigma (must be between 0 and beta)
beta_value = [0.88]  # Update with your desired value for beta (must be between 0 and 1)
psi_value = [2.25]  # Update with your desired value for psi (must be >1)
d_values = [-1, -2, -3, -4]  #Update with your desired values for d (must be <0)
rho_value = [0.01]  #Update with your desired value for rho (must be >0 and <=1)

# Lists to store the intersection cm-values and pm-values for each rho
intersection_cm_values_lists_d = [[] for _ in d_values]
intersection_pm_values_lists_d = [[] for _ in d_values]

# Solve the equation for each combination of values
for sigma in sigma_values:
    for beta in beta_value:
        for rho in rho_value:
            for d_index, d in enumerate(d_values):
                for psi in psi_value:
                    # Solve the equation numerically
                    cm_initial_guess = 0.1  # Initial guess for cm
                    intersection_points2 = fsolve(equation2, cm_initial_guess, args=(psi, beta, sigma, d, rho))

                    # Calculate the corresponding pm values
                    intersection_pm_values = function3(intersection_points2, psi, beta, sigma, d, rho)

                    # Append the intersection cm-values and pm-values to the lists
                    intersection_cm_values_lists_d[d_index].append(intersection_points2)
                    intersection_pm_values_lists_d[d_index].append(intersection_pm_values)

# Set the font properties
font = {'family': 'sans-serif',
        'size': 16,
        'style': 'italic'}

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define marker shapes for different d values
marker_shapes = ['o', 'v', '^', 'D']

# Plot the sigma-values and intersection cm-values for each d
for d_index, d in enumerate(d_values):
    marker = marker_shapes[d_index]
    ax.plot(sigma_values, intersection_cm_values_lists_d[d_index], marker, label='d='+str(d), color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('σ', fontdict=font)
ax.set_ylabel('$c_M$', fontdict=font)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Add legend
ax.legend()

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_8_2.png', dpi=1000)

# Show the plot
plt.show()

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define marker shapes for different d values
marker_shapes = ['o', 'v', '^', 'D']

# Plot the sigma-values and intersection cm-values for each d
for d_index, d in enumerate(d_values):
    marker = marker_shapes[d_index]
    ax.plot(sigma_values, intersection_pm_values_lists_d[d_index], marker, label='d='+str(d), color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('σ', fontdict=font)
ax.set_ylabel('$p_M$', fontdict=font)

# Remove the right and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Move x-axis ticks and label to the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Add legend
ax.legend()

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_9_2.png', dpi=1000)

# Show the plot
plt.show()




# Aggravating


# Equation (4)
def function5(ca, sigma, beta, psi, d, rho):
    if ca < 0:
        return np.inf
    return -1 * ((ca ** (sigma / beta)) / (psi ** (1 / beta)))

# Equation (3)
def function6(ca, sigma, beta, psi, d, rho):
    if ca < 0:
        return np.inf
    return -1 * ((((-1) * (sigma * (ca ** (sigma - 1 - rho)))) / (psi * beta * d * (rho + 1))) ** (1 / (beta - 1)))

# Equation (4) = Equation (3)  
def equation3(ca, sigma, beta, psi, d, rho):
    return function5(ca, sigma, beta, psi, d, rho) - function6(ca, sigma, beta, psi, d, rho)

# Set of predefined values for sigma, beta, psi, d and rho
sigma_values = [0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1]  # Update with your desired values for sigma (must be between beta and infinity)
beta_value = [0.88]  # Update with your desired value for beta (must be between 0 and 1)
psi_value = [2.25]  # Update with your desired value for psi (must be >1)
d_value = [-1]  #Update with your desired value for d (must be <0)
rho_value = [0.01]  #Update with your desired value for rho (must be >0 and <=1)

# Lists to store the intersection c-values and p-values
intersection_ca_values_list = []
intersection_pa_values_list = []

# Solve the equation for each combination of values
for sigma in sigma_values:
    for beta in beta_value:
        for d in d_value:
            for rho in rho_value:
                for psi in psi_value:
                    # Solve the equation numerically
                    ca_initial_guess = 0.1  # Initial guess for ca
                    intersection_points3 = fsolve(equation3, ca_initial_guess, args=(psi, beta, sigma, d, rho))

                    # Calculate the corresponding pa values
                    intersection_pa_values = function5(intersection_points3, psi, beta, sigma, d, rho)

                    # Append the intersection ca-values and pa-values to the lists
                    intersection_ca_values_list.append(intersection_points3)
                    intersection_pa_values_list.append(intersection_pa_values)

# Print the intersection ca-values and pa-values
for i in range(len(intersection_ca_values_list)):
    print("Intersection points for values sigma =", sigma_values[i//len(beta_value)], ", beta =", beta_value[i%len(beta_value)], ", psi =", psi_value[i%len(beta_value)], ", d =", d_value[i%len(beta_value)], " and rho =", rho_value[i%len(beta_value)])
    for j in range(len(intersection_ca_values_list[i])):
        print("ca =", intersection_ca_values_list[i][j])
        print("pa =", intersection_pa_values_list[i][j])
        print()

# Set the font properties
font = {'family': 'sans-serif',
        'size': 16,
        'style': 'italic'}

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the sigma-values and intersection ca-values
ax.plot(sigma_values, intersection_ca_values_list, 'o', color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('σ', fontdict=font)
ax.set_ylabel('$c_A$', fontdict=font)

# Remove the right and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_1_1.png', dpi=1000)

# Show the plot
plt.show()

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Plot the sigma-values and intersection pc-values
ax.plot(sigma_values, intersection_pa_values_list, 'o', color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('σ', fontdict=font)
ax.set_ylabel('$p_A$', fontdict=font)

# Remove the right and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Move x-axis ticks and label to the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_1_2.png', dpi=1000)

# Show the plot
plt.show()




# Aggravating - Robustness checks (different values of rho)


# Equation (4)
def function5(ca, sigma, beta, psi, d, rho):
    if ca < 0:
        return np.inf
    return -1 * ((ca ** (sigma / beta)) / (psi ** (1 / beta)))

# Equation (3)
def function6(ca, sigma, beta, psi, d, rho):
    if ca < 0:
        return np.inf
    return -1 * ((((-1) * (sigma * (ca ** (sigma - 1 - rho)))) / (psi * beta * d * (rho + 1))) ** (1 / (beta - 1)))

# Equation (4) = Equation (3)
def equation3(ca, sigma, beta, psi, d, rho):
    return function5(ca, sigma, beta, psi, d, rho) - function6(ca, sigma, beta, psi, d, rho)

# Set of predefined values for sigma, beta, psi, d and rho
sigma_values = [0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1]  # Update with your desired values for sigma (must be between beta and infinity)
beta_value = [0.88]  # Update with your desired value for beta
psi_value = [2.25]  # Update with your desired value for psi (must be >1)
d_value = [-1]  #Update with your desired value for d (must be <0)
rho_values = [0.01, 1]  #Update with your desired values for rho (must be >0 and <=1)

# Lists to store the intersection ca-values and pa-values for each rho
intersection_ca_values_lists_rho = [[] for _ in rho_values]
intersection_pa_values_lists_rho = [[] for _ in rho_values]

# Solve the equation for each combination of values
for sigma in sigma_values:
    for beta in beta_value:
        for d in d_value:
            for rho_index, rho in enumerate(rho_values):
                for psi in psi_value:
                    # Solve the equation numerically
                    ca_initial_guess = 0.1  # Initial guess for ca
                    intersection_points3 = fsolve(equation3, ca_initial_guess, args=(psi, beta, sigma, d, rho))

                    # Calculate the corresponding pa values
                    intersection_pa_values = function5(intersection_points3, psi, beta, sigma, d, rho)

                    # Append the intersection ca-values and pa-values to the lists
                    intersection_ca_values_lists_rho[rho_index].append(intersection_points3)
                    intersection_pa_values_lists_rho[rho_index].append(intersection_pa_values)

# Set the font properties
font = {'family': 'sans-serif',
        'size': 16,
        'style': 'italic'}

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define marker shapes for different rho values
marker_shapes = ['o', 'v', '^', 'D']

# Plot the sigma-values and intersection ca-values for each rho
for rho_index, rho in enumerate(rho_values):
    marker = marker_shapes[rho_index]
    ax.plot(sigma_values, intersection_ca_values_lists_rho[rho_index], marker, label='ρ='+str(rho), color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('σ', fontdict=font)
ax.set_ylabel('$c_A$', fontdict=font)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Add legend
ax.legend()

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_2_1.png', dpi=1000)

# Show the plot
plt.show()

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define marker shapes for different rho values
marker_shapes = ['o', 'v', '^', 'D']

# Plot the sigma-values and intersection ca-values for each rho
for rho_index, rho in enumerate(rho_values):
    marker = marker_shapes[rho_index]
    ax.plot(sigma_values, intersection_pa_values_lists_rho[rho_index], marker, label='ρ='+str(rho), color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('σ', fontdict=font)
ax.set_ylabel('$p_A$', fontdict=font)

# Remove the right and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Move x-axis ticks and label to the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Add legend
ax.legend()

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_3_1.png', dpi=1000)

# Show the plot
plt.show()




# Aggravating - Robustness checks (different values of d)


# Equation (4)
def function5(ca, sigma, beta, psi, d, rho):
    if ca < 0:
        return np.inf
    return -1 * ((ca ** (sigma / beta)) / (psi ** (1 / beta)))

# Equation (3)
def function6(ca, sigma, beta, psi, d, rho):
    if ca < 0:
        return np.inf
    return -1 * ((((-1) * (sigma * (ca ** (sigma - 1 - rho)))) / (psi * beta * d * (rho + 1))) ** (1 / (beta - 1)))

# Equation (4) = Equation (3)  
def equation3(ca, sigma, beta, psi, d, rho):
    return function5(ca, sigma, beta, psi, d, rho) - function6(ca, sigma, beta, psi, d, rho)

# Set of predefined values for sigma, beta, psi, d and rho
sigma_values = [0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1, 1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.09, 1.1]  # Update with your desired values for sigma (must be between beta and infinity)
beta_value = [0.88]  # Update with your desired value for beta (must be between 0 and 1)
psi_value = [2.25]  # Update with your desired value for psi (must be >1)
d_values = [-1, -2, -3, -4]  #Update with your desired values for d (must be <0)
rho_value = [0.01]  #Update with your desired value for rho (must be >0 and <=1)

# Lists to store the intersection ca-values and pa-values for each d
intersection_ca_values_lists_d = [[] for _ in d_values]
intersection_pa_values_lists_d = [[] for _ in d_values]

# Solve the equation for each combination of values
for sigma in sigma_values:
    for beta in beta_value:
        for rho in rho_value:
            for d_index, d in enumerate(d_values):
                for psi in psi_value:
                    # Solve the equation numerically
                    ca_initial_guess = 0.1  # Initial guess for ca
                    intersection_points3 = fsolve(equation3, ca_initial_guess, args=(psi, beta, sigma, d, rho))

                    # Calculate the corresponding pa values
                    intersection_pa_values = function5(intersection_points3, psi, beta, sigma, d, rho)

                    # Append the intersection ca-values and pa-values to the lists
                    intersection_ca_values_lists_d[d_index].append(intersection_points3)
                    intersection_pa_values_lists_d[d_index].append(intersection_pa_values)

# Set the font properties
font = {'family': 'sans-serif',
        'size': 16,
        'style': 'italic'}

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define marker shapes for different d values
marker_shapes = ['o', 'v', '^', 'D']

# Plot the sigma-values and intersection ca-values for each d
for d_index, d in enumerate(d_values):
    marker = marker_shapes[d_index]
    ax.plot(sigma_values, intersection_ca_values_lists_d[d_index], marker, label='d='+str(d), color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('σ', fontdict=font)
ax.set_ylabel('$c_A$', fontdict=font)

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Add legend
ax.legend()

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_2_2.png', dpi=1000)

# Show the plot
plt.show()

# Create the plot
fig, ax = plt.subplots(figsize=(6, 6))

# Define marker shapes for different d values
marker_shapes = ['o', 'v', '^', 'D']

# Plot the sigma-values and intersection ca-values for each d
for d_index, d in enumerate(d_values):
    marker = marker_shapes[d_index]
    ax.plot(sigma_values, intersection_pa_values_lists_d[d_index], marker, label='d='+str(d), color='black', alpha=0.8)

# Add labels and title
ax.set_xlabel('σ', fontdict=font)
ax.set_ylabel('$p_A$', fontdict=font)

# Remove the right and bottom spines
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)

# Move x-axis ticks and label to the top
ax.xaxis.set_ticks_position('top')
ax.xaxis.set_label_position('top')

# Add legend
ax.legend()

# Adjust subplot spacing
plt.tight_layout()

# Save the plot with DPI=1000
plt.savefig('Figure_3_2.png', dpi=1000)

# Show the plot
plt.show()