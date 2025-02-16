using LinearAlgebra, SparseArrays, Plots, LaTeXStrings

#1D impact and rebound of a rod

#Physical parameters
#(silicon rubber made 100 times less rigid)
rho = 1100 #Density of the rod in Kg/m^3
E = 1.5E7 #Young's modulus of the rod in N/m^2
L = .1 #Length of the rod in metres
alpha = .1 #Damping factor of the rod in Kg/(ms)
#(equivalent to dynamic viscosity)

A = .0001 #Cross-sectional area in m^2 (not given, but needed)
V_0 = sqrt(2*9.8*1) #Initial speed in metres per second
g = 9.8 #gravity in m/s^2

#Numerical parameters
n_z = 4000 #number of segments defined on the rod

#ideally use multiple of 100 (for plotting)
n_t = 50 #number of time intervals that fit in one Unit_time
n_steps = 6*n_t #initial choice of number of time-steps

#Base Units
Unit_length = L
Unit_speed = sqrt(E/rho)
Unit_mass = rho*A*L

#Derived units
Unit_time = Unit_length/Unit_speed
Unit_force = Unit_mass*Unit_speed^2/Unit_length
Unit_density = Unit_mass/(Unit_length^3)

#Time step
delta_t = 1/n_t
delta_z = 1/n_z

#Dimensionless numbers
DD = alpha/(rho*Unit_speed^2*Unit_time) #Dimensionless damping
GG = g*Unit_length/(Unit_speed^2)
UU = V_0/Unit_speed

#Variables
eta = zeros(n_z+1,n_steps+1) #Location of points
w = zeros(n_z+1,n_steps+1) #Velocity of displacement
f = zeros(1,n_steps+1) #Force at the contact point

#Initial values
eta[:, 1] = range(0, stop=1, length=n_z+1)
w[:,1] .= -UU # Julia uses in-place broadcasting for efficiency

#Matrices
# What is the purpose of each matrix?
Mat_A = I(n_z+1) # represents the identity operation, one with no change. Corresponds to the part of the matrix with all the possible displacements of the rod   
Mat_B = -delta_t * I(n_z+1) # represents the time stepping factor; we are propagating the velocity (or displacement since they have the same dimension, backwards in time), since we are using the backward euler method

# Matrices C and D work with the object along its length
Mat_C = -(delta_t / delta_z^2) * # represents the second derivative in terms of the displacement (or the acceleration of each point in the rod)
    (-2 * Diagonal(ones(n_z+1))  # Main diagonal
    + diagm(1 => ones(n_z),      # First superdiagonal
    -1 => ones(n_z)))             # First subdiagonal 

Mat_C[1,2] *= 2 
Mat_C[end,end-1] *= 2


Mat_D = I(n_z+1) - DD * (delta_t / delta_z^2) * # also represents the second derivative of displacement, but this time incorporating damping effects, which reduces the amplitude of the strain wave on the rod
        (-2 * Diagonal(ones(n_z+1)) # creates the main diagonal
        + diagm(1 => ones(n_z), # creates the first superdiagonal
        -1 => ones(n_z))) # creates the first subdiagonal 
Mat_D[1,2] *= 2
Mat_D[end,end-1] *= 2

Mat_E = zeros(n_z+1,1) 
Mat_E[1] = -2*DD/delta_z - 2*delta_t/delta_z # Additional vector that will alter the system of equations based on the presence of an external force; what will essentially apply the strain wave over the object over the boundaries

Mat_System = [Mat_A Mat_B zeros(n_z+1,1); Mat_C Mat_D Mat_E]

Mat_System = sparse(Mat_System)

#Main loop
for ind_time in 1:n_steps
    println(ind_time/n_steps)

    b = [eta[:, ind_time]; w[:, ind_time] - GG * delta_t * ones(n_z+1)] 
    b[n_z+2] -= 2 * DD * f[ind_time] * delta_z / delta_t - 2 * delta_t / delta_z
    b[2*n_z+2] += 2 * delta_t / delta_z

    println("Mat_System size: ", size(Mat_System))
    println("b size: ", size(b))
    println("Submatrix size: ", size(Mat_System[1:end, 1:end-1]))

    #First we solve without force
    sol_free = Mat_System[1:end,1:end-1]\b # Is the solution vector that does not include the force term f[ind_time], which represents the rod's behavior when there is no external force 
    # We remove the last column because it corresponds to the external forces applied to the system and we are looking to simulate a body in free motion
    # the situation above essentially involves the free motion of the rod
    # the dimension of this vector will be 2*n_z + 2 because the first n_z + 1 elements corresponds to the displacement of each of the n_z + 1 spatial points on the rod at given step
    # the other n_z + 1 elements are the velocity of each of the n_z + 1 spatial points on the rod at the same time step

    # Elements from sol_free[1:n_z + 1] represent the displacement of the rod at the current time step (represents each eta)
    # Elements from sol_free[n_2 + 2: 2*n_z + 2] represents the velocity of the rod at the current time step (represents each w in the code)
    # Example n_z = 3, we will have a sol_free vector of 8 elements: sol_free = [eta_1, eta_2, eta_3, eta_4, w_1, w_2, w_3, w_4]
    
    # After solving for sol_free, we check the first element of system to determine if a force needs to be applied because there is a break from inertia, or the free movement of the body 
    if sol_free[1] < 0 # sol_free[1] is the displacement at the first point of the rod 

        # If there is an impact, we recompute the solution with forces for the next step of the simulations
        sol_forced = Mat_System[2:end,[2:n_z+1,n_z+3:end]]\b[2:end] # solves the system which considers the force back into the MatSystem
        eta[1,ind_time+1] = 0 # set the the velocity and displacement of the rod section to zero after the impact
        w[1,ind_time+1] = 0
        eta[2:end,ind_time+1] = sol_forced[1:n_z] # the displacement and velocity of the remaining points are updated based on the new solution
        w[2:end,ind_time+1] = sol_forced[n_z+1:2*n_z]
        f[ind_time+1] = sol_forced[end] # force at the contact point is also updated after the impact  
    else
        eta[:,ind_time+1] = sol_free[1:n_z+1] # update the velocity and displacement based on the free motion
        w[:,ind_time+1] = sol_free[n_z+2:2*n_z+2]
    end
end

# Plotting rod points
p1 = plot(delta_t .* (0:n_steps), eta[1, :], label="Leading edge", linewidth=2, color=:blue)
plot!(delta_t .* (0:n_steps), eta[end, :], label="Trailing edge", linewidth=2, color=:red)
for ind_z in 101:100:n_z
    plot!(delta_t .* (0:n_steps), eta[ind_z, :], linewidth=1, color=:black)
end
plot!([0, 1, 1, 2], [0, 1, 1, 0], color=RGB(0.5, 0.5, 0.5), linewidth=2, label="Wave front")
xlabel!(L"tC/L")
ylabel!(L"\frac{z}{L}")
title!("Rod Motion")
savefig("rod_motion.png")

# Plotting forces
p2 = plot(delta_t .* (0:n_steps), f[1, :] .* Unit_force ./ (V_0 * rho * A * Unit_speed), linewidth=2)
xlabel!(L"tC/L")
ylabel!(L"\frac{f}{\rho A C V_0}")
title!("Impact Forces")
savefig("force.png")

# Strain calculation
strain = zeros(size(eta))
for ind_time in 1:n_steps+1
    strain[1, ind_time] = f[ind_time] * Unit_force / (E * A)
    for ind_z in 2:n_z
        strain[ind_z, ind_time] = 1 - (eta[ind_z+1, ind_time] - eta[ind_z-1, ind_time]) / (2 * delta_z)
    end
end

# Plotting strain
p3 = heatmap(delta_t .* (0:n_steps), delta_z .* (0:n_z), strain * E / (rho * Unit_speed * V_0), color=:cool)
xlabel!(L"tC/L")
ylabel!(L"z/L")
title!("Strain")
savefig("strain.png")

# Maximum Strain
println("Maximum Strain: ", maximum(strain * E / (rho * Unit_speed * V_0)))

