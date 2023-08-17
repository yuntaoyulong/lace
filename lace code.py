import os
import gdsfactory as gf
import numpy as np
from PySpice.Spice.Netlist import Circuit
# from PySpice.Unit import *
import PySpice.Logging.Logging as Logging
import networkx as nx;
# import matplotlib.pyplot as plt
import gc
import time
import mlrose
# from numba import jit
import gc
import sys
from PySpice.Spice.NgSpice.Shared import NgSpiceShared
# from cffi import FFI
# from memory_profiler import profile
NgSpiceShared.NUMBER_OF_EXEC_CALLS_TO_RELEASE_MEMORY = 20
# from numba import jit
L= [11.0,3.0,1.0,-1.0,-3.0,-3.0]
Number_heater=6
heater_length = 5
heater_width = 1
width = []
x_coords = []
padground_center =(100,9)
pad_center = []
heater_center = []
for i in range(Number_heater):
    pad_center.append((-30+i*4,9))
heater_center.append((10,0))
heater_center.append((2,-4))
heater_center.append( (-0,-8))
heater_center.append((-2,-12))
heater_center.append((-4,-16))
heater_center.append((-4,-20))
width_contant=2
pad_size = (4,3)
pad_layer = (5,0)
contact_length = 1.5 
PDK = gf.generic_tech.get_generic_pdk()

PDK.activate()
for i in range(6):
    width.append(np.random.uniform(0, 2))
    x_coords.append(np.random.uniform(0, 2))
Number_heater=6
parameter = width + x_coords
pad_reference = gf.components.rectangle(size=pad_size,layer=pad_layer).rotate(90)
pad_reference.add_port("o1", center=(pad_reference.ports["e1"].x,pad_reference.ports["e1"].y+contact_length ), orientation=270, width=0, layer=(5,0))
heater_reference = gf.components.rectangle(size=(heater_length,heater_width),layer=(11,0))
heater_reference.add_port("o1", center=(heater_reference.ports["e1"].x + contact_length,heater_reference.ports["e1"].y), orientation=180, width=0, layer=(11,0))
heater_reference.add_port("o2", center=(heater_reference.ports["e3"].x - contact_length,heater_reference.ports["e3"].y), orientation=0, width=0, layer=(11,0))
def interface(parameter):
    width = [round(x, 3) for x in parameter[:6]]


    c = gf.Component('nihao')
    gf.config.rich_output()
    
    
    pad= []
    padground = [] 
   
    for i in range(Number_heater):
        pad.append(c << pad_reference)
        pad[i].center= pad_center[i]

    padground.append(c << pad_reference)
    padground[0].center = padground_center
    heater = []
    
    
    


        
    for i in range(Number_heater):
        heater.append(c << heater_reference)
        heater[i].center = heater_center[i]
    
    layer=12
    for i in range(6):
        x_coords[i]=round(parameter[i+6]*((padground_center[0]-heater[i].center[0]-heater_width)/2 -2)+ heater[i].center[0] + heater_width/2 +1,3)
    right_ports = [
        gf.Port(f"R_{i}", center=(heater[i].ports['o1'].center[0], heater[i].ports['o1'].center[1]), width=0.5, orientation=180, layer=layer)        for i in range(Number_heater)
    ]
    left_ports = [
        gf.Port(f"L_{i}", center=( pad[i].ports['o1'].center[0], pad[i].ports['o1'].center[1]), width=0.5, orientation=270, layer=layer)
        for i in range(Number_heater)
    ]

    # you can also mess up the port order and it will sort them by default
    left_ports.reverse()

    routes = gf.routing.get_bundle(
        left_ports, right_ports, sort_ports=True,radius=None,layer=(12,0),width=width_contant,start_straight_length=0,separation=0
    ) 

    for route in routes:
        c.add(route.references)




    route_small = []
    route = [route_small for i in range(Number_heater)]

    route[0] = gf.routing.get_route(padground[0].ports["o1"],heater[0].ports["o2"], width=width[0],radius=None,layer=(12,0))

    
    c.add(route[0].references)

    rand = gf.Component()
    np.set_printoptions(precision = 2)

    y_coords = [padground[0].ports["o1"].y]
    x_coords[0] = padground[0].ports["o1"].x
    layer=12
    j = []
    max_length_layer ={}
    max_length_layer[0] = (padground[0].ports["o1"].x)

    Resistance_each_layer = [route_small for i in range(Number_heater)]
    small_layer = [0] *Number_heater
    smaller_layer = [0] *Number_heater
    G = nx.Graph()
    pos = nx.spring_layout(G)
    Resistance_each_layer[0] = (padground[0].ports["o1"].x-heater[0].ports["o2"].x +padground[0].ports["o1"].y-heater[0].ports["o2"].y)/width[0]
    Resistance_left_position = []
    Resistance_left_position_occupation = []

    for i in range(Number_heater):
        Resistance_left_position.append(heater[i].ports["o2"].x)
        Resistance_left_position_occupation.append(i)
    # print(Resistance_left_position)

    G.add_weighted_edges_from([('P'+str(0)*2, 'P0-1',Resistance_each_layer [0])])
    # print(type(max_length_layer))
    vertical_line =[route_small for i in range(Number_heater)]
    vertical_line[1] =0 
    for i in range(1,Number_heater):
        j = np.random.randint(1,len(route[i-1].references))
        if i > 1: 
            port = 'o'
            layer= 12
            # print(route[i-1].references)
            if route[i-1].references[2].ports[port + str(1)].x == route[i-1].references[2].ports[port + str(2)].x:
                vertical_line[i-1]  = 0
                # print(max_length_layer)
                max_length_layer = dict(sorted(max_length_layer.items(), key=lambda x: x[1], reverse=False))

                max_length_layer = {k: v for k, v in  max_length_layer.items() if v >= x_coords[i]}
                
                t=0
                for key,value in max_length_layer.items():
                    t+=1
                    if t == 1:
                        small_layer[i]= key 
                        #print(smaller_layer[i],max_length_layer,'114514')
                        if len(max_length_layer) == 1:
                            smaller_layer[i] = -1
                    if t == 2:
                        if key == 0  and small_layer[i] == 0:
                            key= key - 1
                        smaller_layer[i]= key
                        break
                if small_layer[i] == 0:
                    y_coords.append(route[small_layer[i]].references[vertical_line[i-1]+1].ports[port + str(2)].y  )
                else: 
                    y_coords.append(route[small_layer[i]].references[vertical_line[i-1]+1].ports[port + str(2)].y + width[small_layer[i]]/2)
                max_length_layer[i] = x_coords[i]

                
                G.remove_edge('P{}{}'.format( Resistance_left_position_occupation[small_layer[i]],small_layer[i]) ,'P{}{}'.format(small_layer[i],smaller_layer[i]))
                
                #print(i,'delele','P{}{}'.format(Resistance_left_position_occupation[small_layer[i]],small_layer[i]),'P{}{}'.format(small_layer[i],smaller_layer[i]))
                G.add_edge('P{}{}'.format(Resistance_left_position_occupation[small_layer[i]],small_layer[i]), 'P{}{}'.format(i,small_layer[i]), weight=(-Resistance_left_position[small_layer[i]]+x_coords[i])/width[small_layer[i]])
                #print('P{}{}'.format(Resistance_left_position_occupation[small_layer[i]],small_layer[i]), 'P{}{}'.format(i,small_layer[i]))
                #print((-Resistance_left_position[small_layer[i]]+x_coords[i])/width[small_layer[i]],'qwer')
                Resistance_each_layer[small_layer[i]] = Resistance_each_layer[small_layer[i]] + (Resistance_left_position[i]-x_coords[i])/width[small_layer[i]]
                G.add_edge('P{}{}'.format(small_layer[i],smaller_layer[i]), 'P{}{}'.format(i,small_layer[i]), weight=Resistance_each_layer[i-1])
                Resistance_each_layer[i] = (x_coords[i]-Resistance_left_position[i] +y_coords[i]-heater[i].ports["o2"].y)/width[i]
                #print('P{}{}'.format(small_layer[i],smaller_layer[i]), 'P{}{}'.format(i,small_layer[i]),Resistance_each_layer[i-1])
                G.add_edge('P{}{}'.format(i,i), 'P{}{}'.format(i,small_layer[i]), weight=Resistance_each_layer[i])
                #print('P{}{}'.format(i,i), 'P{}{}'.format(i,small_layer[i]),Resistance_each_layer[i])
                Resistance_left_position[small_layer[i]] = x_coords[i]
                Resistance_left_position_occupation[small_layer[i]] = i
                orientation = 270
                end_straight_length=0
                layer = 8


            else:  
                vertical_line[i-1]  = 1
                max_length_layer = dict(sorted(max_length_layer.items(), key=lambda x: x[1], reverse=False))
                # print(route[list(max_length_layer.keys())[0]]) 
                max_length_layer = {k: v for k, v in  max_length_layer.items() if v >= x_coords[i]}
                t=0
                # print(max_length_layer)
                for key,value in max_length_layer.items():
                    t+=1
                    if t == 1:
                        small_layer[i]= key 
                       # print(smaller_layer[i],max_length_layer,'114514')
                        if len(max_length_layer) == 1:
                            smaller_layer[i] = -1



                    if t == 2:
                        if key == 0 and small_layer[i] == 0:
                            key= key - 1
                        smaller_layer[i]= key# print(small_layer[i])
                        # print(smaller_layer[i],'hsiohdadkfg')
                        break
                if small_layer[i] == 0:
                    y_coords.append(route[small_layer[i]].references[vertical_line[i-1]+1].ports[port + str(2)].y  )
                else: 
                    y_coords.append(route[small_layer[i]].references[vertical_line[i-1]+1].ports[port + str(2)].y + width[small_layer[i]]/2)
                max_length_layer[i] = x_coords[i]

                G.remove_edge('P{}{}'.format( Resistance_left_position_occupation[small_layer[i]],small_layer[i]) ,'P{}{}'.format(small_layer[i],smaller_layer[i]))
                
                #print(i,'delele','P{}{}'.format(Resistance_left_position_occupation[small_layer[i]],small_layer[i]),'P{}{}'.format(small_layer[i],smaller_layer[i]))
                G.add_edge('P{}{}'.format(Resistance_left_position_occupation[small_layer[i]],small_layer[i]), 'P{}{}'.format(i,small_layer[i]), weight=(-Resistance_left_position[small_layer[i]]+x_coords[i])/width[small_layer[i]])
                #print('P{}{}'.format(Resistance_left_position_occupation[small_layer[i]],small_layer[i]), 'P{}{}'.format(i,small_layer[i]))
                #print((-Resistance_left_position[small_layer[i]]+x_coords[i])/width[small_layer[i]],'qwer')
                Resistance_each_layer[small_layer[i]] = Resistance_each_layer[small_layer[i]] + (Resistance_left_position[i]-x_coords[i])/width[small_layer[i]]
                G.add_edge('P{}{}'.format(small_layer[i],smaller_layer[i]), 'P{}{}'.format(i,small_layer[i]), weight=Resistance_each_layer[i-1])
                Resistance_each_layer[i] = (x_coords[i]-Resistance_left_position[i] +y_coords[i]-heater[i].ports["o2"].y)/width[i]
                #print('P{}{}'.format(small_layer[i],smaller_layer[i]), 'P{}{}'.format(i,small_layer[i]),Resistance_each_layer[i-1])
                G.add_edge('P{}{}'.format(i,i), 'P{}{}'.format(i,small_layer[i]), weight=Resistance_each_layer[i])
                #print('P{}{}'.format(i,i), 'P{}{}'.format(i,small_layer[i]),Resistance_each_layer[i])
                Resistance_left_position[small_layer[i]] = x_coords[i]
                Resistance_left_position_occupation[small_layer[i]] = i

                orientation = 270
                end_straight_length=0
                layer = 7


        else:
                j = 1 
                G.remove_edge('P00', 'P0-1')
                port = 'o'
                max_length_layer[i] = x_coords[i]
                y_coords.append(route[i-1].references[2].ports[port + str(2)].y +  width[0]/2)
                orientation = 270
                end_straight_length=0
                
                layer = 31
                # print(x_coords,y_coords,111)
                # print(route[i-1].references[2].rotation)
                G.add_edge('P00', 'P10', weight=-1 * (heater[0].ports["o2"].x-x_coords[i])/width[0])
                
                Resistance_each_layer[0] = Resistance_each_layer[0] + (heater[0].ports["o2"].x-x_coords[1])/width[0] 
                G.add_edge('P10', 'P0-1', weight=Resistance_each_layer[0])
                Resistance_each_layer[1] = (x_coords[1]-heater[1].ports["o2"].x +y_coords[i]-heater[1].ports["o2"].y)/width[1]
                
                small_layer[0]=-1 
                small_layer[1]= 0       
                Resistance_left_position_occupation[0] = 1                                                                
                G.add_edge('P11', 'P10', weight=Resistance_each_layer[1])
                # print((-Resistance_left_position[i]+x_coords[i])/width[i-1],Resistance_each_layer[i-1],Resistance_each_layer[i])

        rand.add_port(name="j" + str(i), center=(x_coords[i],y_coords[i]), width=2, orientation=orientation,layer=layer)
        route[i]=gf.routing.get_route(heater[i].ports["o2"], rand.ports["j" + str(i)], width=width[i],radius=None,layer=(layer,0),end_straight_length=end_straight_length)
        c.add(route[i].references)
        # c.plot_matplotlib()
    del(c)
    print('x',x_coords,'w',width,file=open(os.path.join(".", 'cansh.txt'), 'a'))
    #c.plot_matplotlib()
    #display(c)
    # print(small_layer)
    #plt.figure()
    #pos = nx.spring_layout(G)
    # nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # plt.show()
    # print(c)
    #c.write_gds("init.gds",with_metadata=True)
    # A=nx.adjacency_matrix(G).todense()
    # print(A)
    Resistor_Left = []
    Resistor_heater = []
    for i in range(Number_heater):
        Resistor_Left.append((heater[i].ports['o1'].x-pad[i].ports['o1'].x+pad[i].ports['o1'].y-heater[i].ports['o1'].y)/width_contant)
        Resistor_heater.append(heater_length/heater_width*0.485/ 36.6e-3)
        G.add_edge('P{}{}'.format(i,i), 'P{}L'.format(i), weight=Resistor_heater[i])
        G.add_edge('P{}L'.format(i), 'P{}Vcc'.format(i), weight=Resistor_Left[i])
    
    # print(Resistor_Left,Resistor_heater)
    # plt.figure()
    #pos = nx.spring_layout(G)
    #nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    #nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    # plt.show()
    # display(labels)
    return circuit(labels)
# @profile
def circuit(labels):
    print('gc print 1')
    gc.collect()
    print('gc print 2')
    current_minute = time.localtime().tm_min
    current_second = time.localtime().tm_sec

    # 打印分钟和秒钟
    print("当前时间：{}分{}秒".format(current_minute, current_second,'medium'))
    logger = Logging.setup_logging()
    minl = 1e-6
    sigma = 0
    sigma_cross = 0
    #NgSpiceShared.setup_platform()
    for j in range(200):
        circuit = Circuit('Resistor Bridge')
        circuit_diff = []
        for i in range(Number_heater):
            circuit_diff.append(Circuit('Resistor Bridge[{}]'.format(i)))
        V = [5] * Number_heater
        for i in range(Number_heater):
            V[i] = np.random.uniform(0, 5)
            circuit.V('pad{}'.format(i), '{}Vcc'.format(i), circuit.gnd, V[i] )
            for k in range(Number_heater):
                if k != i:
                    circuit_diff[k].V('pad{}'.format(i), '{}Vcc'.format(i), circuit_diff[k].gnd, V[i] )
                else:
                    circuit_diff[k].V('pad{}'.format(i), '{}Vcc'.format(i), circuit_diff[k].gnd, (V[i]+minl) )
                    circuit_diff[k].V('gnd', '0-1', circuit_diff[k].gnd, 0 )
            
        circuit.V('gnd', '0-1', circuit.gnd, 0 )    
        i = 0
        for  key,value in labels.items():
            # print((key[0][1:]),(key[1][1:]),value)
            circuit.R(i,(key[0][1:]),(key[1][1:]),value )
            for k in range(Number_heater):
                circuit_diff[k].R(i,(key[0][1:]),(key[1][1:]),value )
            i+=1
        simulator_diff = []
        analysis_diff = [] 
        simulator = circuit.simulator(temperature=25, nominal_temperature=25)
        analysis = simulator.operating_point()
        result = {}
        result_diff = []
        Voltage_diff = []
        Voltage_pad_diff = []
        for k in range(Number_heater):
            simulator_diff.append(circuit_diff[k].simulator(temperature=25, nominal_temperature=25))
            data = simulator_diff[k].operating_point()
            
            analysis_diff.append(data)
            result_diff.append({})
            Voltage_diff.append([])
            Voltage_pad_diff.append([])
        Voltage = []
        Voltage_pad = []
        for k in range(Number_heater):
            for node in analysis_diff[k].nodes.values():
                result_diff[k][str(node)]=float(node)
        for node in analysis.nodes.values():
            result[str(node)]=float(node)# Fixme: format value + unit    
        ngspice = simulator.factory(circuit).ngspice
        ngspice.remove_circuit()
        ngspice.destroy() 
        for k in range(Number_heater):
            ngspice = simulator_diff[k].factory(circuit_diff[k]).ngspice
            ngspice.remove_circuit()
            ngspice.destroy()
            # ngspice.reset()
           # del(simulator_diff[k].operating_point)
            for i in range(Number_heater):
                Voltage_diff[k].append(result_diff[k]['{}l'.format(i)]-result_diff[k]['{}{}'.format(i,i)])
                Voltage_pad_diff[k].append(result_diff[k]['{}vcc'.format(i)])
        for i in range(Number_heater):
            Voltage.append(result['{}l'.format(i)]-result['{}{}'.format(i,i)])
            Voltage_pad.append(result['{}vcc'.format(i)])
            sigma = sigma + abs((Voltage_diff[i][i]-Voltage[i])/(Voltage_pad_diff[i][i]-Voltage_pad[i])*Voltage_pad[i])
            #print(Voltage_diff[i][i],Voltage[i],Voltage_pad_diff[i][i],Voltage_pad[i],(Voltage_diff[i][i]-Voltage[i])/(Voltage_pad_diff[i][i]-Voltage_pad[i])*Voltage_pad[i])
        for k in range(Number_heater):
            for i in range(Number_heater):
                if i != k:
                    sigma_cross = sigma_cross + abs((Voltage_diff[k][i]-Voltage[i])/(Voltage_pad_diff[k][k]-Voltage_pad[k])*Voltage_pad[i])
    sigma_cross_sum= sigma_cross
    sigma_sum= sigma
    del(Voltage_diff,result_diff,analysis_diff,analysis,result,circuit,circuit_diff,data)
    gc.disable()
    gc.collect()

    print('small:',sigma_cross_sum,'big:',sigma_sum,'need:',sigma_sum/sigma_cross_sum,file=open(os.path.join(".", 'groups.txt'), 'a'))

    # 获取当前时间的分钟和秒钟
    current_minute = time.localtime().tm_min
    current_second = time.localtime().tm_sec

    # 打印分钟和秒钟
    print("当前时间：{}分{}秒".format(current_minute, current_second))

    
        
    
    quit
    return (sigma_sum/sigma_cross_sum)

def main():
    print('',file=open(os.path.join(".", 'groups.txt'), 'w'))
    print('',file=open(os.path.join(".", 'cansh.txt'), 'w'))
    print('',file=open(os.path.join(".", 'gros.txt'), 'w'))



    fitness_dist = mlrose.CustomFitness(interface)

    problem_fit = mlrose.ContinuousOpt(length = 12, fitness_fn = fitness_dist, maximize = True, max_val = 2,min_val = 0.1)
    best_tour,cost = mlrose.genetic_alg(problem_fit, mutation_prob = 0.5, max_attempts = 10, random_state = 2, pop_size =100,max_iters=100)

    print(cost, best_tour[:6],[round(best_tour[i+6]*(padground_center[0]/2-heater_center[i][0]/2-1.75)+ heater_center[i][0] + 3.5, 3) for i in range(6)])

main()
