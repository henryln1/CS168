# This code is based on an assignment designed by Paul Valiant.  
# Please do not disseminate this code without his written permission.
#
# This python implementation was written by Okke Schrijvers for CS168



import sys
import math
import numpy as np
import time
from matplotlib import pyplot as plt
from matplotlib import animation
import copy
import random


data = []

# plan is an array of 40 floating point numbers
def sim(plan):
    for i in range(0, len(plan)):
        if plan[i] > 1:
            plan[i] = 1.0
        elif plan[i] < -1:
            plan[i] = -1.0

    dt = 0.1
    friction = 1.0
    gravity = 0.1
    mass = [30, 10, 5, 10, 5, 10]
    edgel = [0.5, 0.5, 0.5, 0.5, 0.9]
    edgesp = [160.0, 180.0, 160.0, 180.0, 160.0]
    edgef = [8.0, 8.0, 8.0, 8.0, 8.0]
    anglessp = [20.0, 20.0, 10.0, 10.0]
    anglesf = [8.0, 8.0, 4.0, 4.0]
    
    edge = [(0, 1),(1, 2),(0, 3),(3, 4),(0, 5)]
    angles = [(4, 0),(4, 2),(0, 1),(2, 3)]
    
    # vel and pos of the body parts, 0 is hip, 5 is head, others are joints
    v = [[0.0,0.0,0.0,0.0,0.0,0.0], [0.0,0.0,0.0,0.0,0.0,0.0]]
    p = [[0, 0, -.25, .25, .25, .15], [1, .5, 0, .5, 0, 1.9]]
    
    spin = 0.0
    maxspin = 0.0
    lastang = 0.0
    
    for j in range(20):
        for k in range(10):
            lamb = 0.05 + 0.1*k
            t0 = 0.5
            if j>0:
                t0 = plan[2*j-2]
            t0 *= (1-lamb)
            t0 += plan[2*j]*lamb
            
            t1 = 0.0
            if j>0:
                t1 = plan[2*j-1]
            t1 *= (1-lamb)
            t1 += plan[2*j+1]*lamb
            
            

            contact = [False,False,False,False,False,False]
            for z in range(6):
                if p[1][z] <= 0:
                    contact[z] = True
                    spin = 0
                    p[1][z] = 0

            anglesl = [-(2.8+t0), -(2.8-t0), -(1-t1)*.9, -(1+t1)*.9]

            disp = [[0,0,0,0,0],[0,0,0,0,0]]
            dist = [0,0,0,0,0]
            dispn = [[0,0,0,0,0],[0,0,0,0,0]]
            for z in range(5):
                disp[0][z] = p[0][edge[z][1]]-p[0][edge[z][0]]
                disp[1][z] = p[1][edge[z][1]]-p[1][edge[z][0]]
                dist[z] = math.sqrt(disp[0][z]*disp[0][z] + disp[1][z]*disp[1][z])
                inv = 1.0/dist[z];
                dispn[0][z] = disp[0][z]*inv
                dispn[1][z] = disp[1][z]*inv;
        
            dispv = [[0,0,0,0,0],[0,0,0,0,0]]
            distv = [0,0,0,0,0]
            for z in range(5):
                dispv[0][z] = v[0][edge[z][1]] - v[0][edge[z][0]]
                dispv[1][z] = v[1][edge[z][1]] - v[1][edge[z][0]]
                distv[z] = 2*(disp[0][z]*dispv[0][z] + disp[1][z]*dispv[1][z])

            
            forceedge = [[0,0,0,0,0],[0,0,0,0,0]]
            for z in range(5):
                c = (edgel[z]-dist[z])*edgesp[z]-distv[z]*edgef[z]
                forceedge[0][z] = c*dispn[0][z]
                forceedge[1][z] = c*dispn[1][z]

            edgeang = [0,0,0,0,0]
            edgeangv = [0,0,0,0,0]
            for z in range(5):
                edgeang[z] = math.atan2(disp[1][z], disp[0][z])
                edgeangv[z] = (dispv[0][z]*disp[1][z]-dispv[1][z]*disp[0][z])/(dist[z]*dist[z])

            inc = edgeang[4] - lastang
            if (inc < -math.pi):
                inc += 2.0 * math.pi
            elif inc > math.pi:
                inc -= 2.0 * math.pi
            spin += inc
            spinc = spin - .005*(k + 10 * j)
            if spinc > maxspin:
                maxspin = spinc
                lastang = edgeang[4]

            angv = [0,0,0,0]
            for z in range(4):
                angv[z] = edgeangv[angles[z][1]]-edgeangv[angles[z][0]];

            angf = [0,0,0,0]
            for z in range(4):
                ang = edgeang[angles[z][1]]-edgeang[angles[z][0]]-anglesl[z]
                if ang > math.pi:
                    ang -= 2*math.pi
                elif ang < -math.pi:
                    ang += 2*math.pi
                m0 = dist[angles[z][0]]/edgel[angles[z][0]]
                m1 = dist[angles[z][1]]/edgel[angles[z][1]]
                angf[z] = ang*anglessp[z]-angv[z]*anglesf[z]*min(m0,m1)

            edgetorque = [[0,0,0,0,0],[0,0,0,0,0]]
            for z in range(5):
                inv = 1.0 / (dist[z]*dist[z])
                edgetorque[0][z] = -disp[1][z]*inv
                edgetorque[1][z] =  disp[0][z]*inv

            for z in range(4):
                i0 = angles[z][0]
                i1 = angles[z][1]
                forceedge[0][i0] += angf[z]*edgetorque[0][i0]
                forceedge[1][i0] += angf[z]*edgetorque[1][i0]
                forceedge[0][i1] -= angf[z]*edgetorque[0][i1]
                forceedge[1][i1] -= angf[z]*edgetorque[1][i1]

            f = [[0,0,0,0,0,0],[0,0,0,0,0,0]]
            for z in range(5):
                i0 = edge[z][0]
                i1 = edge[z][1]
                f[0][i0] -= forceedge[0][z]
                f[1][i0] -= forceedge[1][z]
                f[0][i1] += forceedge[0][z]
                f[1][i1] += forceedge[1][z]

            for z in range(6):
                f[1][z] -= gravity*mass[z]
                invm = 1.0/mass[z]
                v[0][z] += f[0][z]*dt*invm
                v[1][z] += f[1][z]*dt*invm
                
                if contact[z]:
                    fric = 0.0
                    if v[1][z] < 0.0:
                        fric = -v[1][z]
                        v[1][z] = 0.0

                    s = np.sign(v[0][z])
                    if v[0][z]*s < fric*friction:
                        v[0][z]=0
                    else:
                        v[0][z] -= fric*friction*s
                p[0][z] += v[0][z] * dt
                p[1][z] += v[1][z] * dt;

            data.append(copy.deepcopy(p))


            if contact[0] or contact[5]:
                return p[0][5]
    return p[0][5]


###########
# The following code is given as an example to store a video of the run and to display
# the run in a graphics window. You will treat sim(plan) as a black box objective
# function and maximize it.
###########

def sim_wrapper(plan):
    return - sim(plan)
plan = [random.uniform(-1,1) for i in range(40)]

MCMC = True



if MCMC: 
    max_iterations = 3000


    #MCMC algorithm

    best_plan = plan


    best_plan = plan
    furthest_distance = sim(plan)

    T = 10

    replace_number = 4
    print_every = 500
    number_trials = 50
    best_distance = float('-infinity')
    best_plan_overall_all_trials = []

    for g in range(number_trials):
        print("Trial: ", g)
        for i in range(max_iterations):
            #furthest_distance = float('-infinity')
            if i % print_every == 0 :
                print("Iteration: ", i)
            #random_index = random.randint(0, len(plan) - 1)
            #random_index_two = random.randint(0, len(plan) - 1)
            possible_new_plan = best_plan.copy()
            #possible_new_plan[random_index] = random.uniform(-1 , 1)
            #possible_new_plan[random_index_two] = random.uniform(-1 , 1)
            for i in range(replace_number):
                random_index = random.randint(0, len(plan) - 1)
                possible_new_plan[random_index] = random.uniform(-1 , 1)
            change_distance_traveled = sim(possible_new_plan) - sim(plan)
            if change_distance_traveled < 0 or (T > 10 and random.uniform(0, 1) < math.exp( - change_distance_traveled / T)):
                plan = possible_new_plan
            if sim(possible_new_plan) > furthest_distance:
                best_plan = possible_new_plan
                furthest_distance = sim(possible_new_plan)
        print(best_plan)
        print("This trial's furthest distance: ", furthest_distance)
        if furthest_distance > best_distance:
            best_distance = furthest_distance
            best_plan_overall_all_trials = best_plan



    print("Furthest Distance reached over all trials: ", best_distance)

    data = []

    sim(best_plan_overall_all_trials)


print("best plan post MCMC: ", best_plan_overall_all_trials)
#try scipy.optimize.minimize to try and solve this. Remember to do a - sign so that it will actually maxximize distance

#else: #Do gradient descent algorithm

# learning_rate = 0.01
# plan = [random.uniform(-1,1) for i in range(40)]
# plan = best_plan_overall_all_trials
# gradient_descent_num_iterations = 3000
# print_every = 100
# for i in range(gradient_descent_num_iterations):
#     data = []
#     shifted_plan = [x - 0.05 for x in plan]
#     if i % 100 == 0:
#         print ("Gradient Descent Iteration: ", i)
#         print("Max Distance Achieved: ", sim(plan))

#     distance = sim(plan)
#     shifted_distance = sim(shifted_plan)
#     numerical_gradient = (shifted_distance - distance) / 2

#     numerical_gradient = learning_rate * numerical_gradient

#     new_plan = [x - numerical_gradient for x in plan]
#     plan = new_plan

plan = best_plan_overall_all_trials


from scipy.optimize import minimize, rosen, rosen_der

print("Scipy")
results = minimize(sim_wrapper, plan)
print(results)
best_option = results.x
print("Best Distance: ", sim(best_option))
#exit()


data = []
sim(best_option)

print("best plan after scipy: ", best_option)


# draw the simulation
fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(12, 3)

ax = plt.axes(xlim=(-1, 10), ylim=(0, 3))

joints = [5, 0, 1, 2, 1, 0, 3, 4]
patch = plt.Polygon([[0,0],[0,0]],closed=None, fill=None, edgecolor='k')
head = plt.Circle((0, 0), radius=0.15, fc='k', ec='k')

def init():
    ax.add_patch(patch)
    ax.add_patch(head)
    return patch,head


def animate(j):
    points = zip([data[j][0][i] for i in joints], [data[j][1][i] for i in joints])
    patch.set_xy(list(points))
    head.center = (data[j][0][5], data[j][1][5])
    return patch,head

anim = animation.FuncAnimation(fig, animate,
                               init_func=init,
                               frames=len(data),
                               interval=20)

#anim.to_html5_video()
anim.save('animation.mp4', fps=50)

plt.show()