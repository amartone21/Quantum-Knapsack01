import random
from qiskit import *
import numpy as np
from deap import tools, base, creator
import pandas as pd
import matplotlib.pyplot as plt
import time
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeGuadalupe

'''

NOTE: The script saves the final graph in a folder called Graphs which was already present on my device,
In any case I think without the folder there is a problem saving the graph.
Unfortunately I wrote it in italian so maybe no everything is clear. I will force myself to translate everything one day.


'''

# Define  DATASET
#   name    weight    value
oggetti = pd.DataFrame.from_records([
    ['compass', 70, 135],
    ['water', 73, 139],
    ['sandwich', 77, 149],
    ['glucose', 80, 150],
    ['tin', 82, 156],
    ['banana', 87, 163],
    ['apple', 90, 173],
    ['cheese', 94, 184],
    ['beer', 98, 192],
    ['suntan cream', 106, 201],
    ['camera', 110, 210],
    ['T-shirt', 113, 214],
    ['trousers', 115, 221],
    ['umbrella', 118, 229],
    ['note-case', 120, 240]])

oggetti.columns = ['Nome', 'Peso', 'Valore']
lunghezza_cromosoma = len(oggetti['Nome'])

#definition of constraint and parametri
capienza = 750
n_popolazione = 30

n_generazioni = 30
numero_semi = 50

# optimal values obtained from tuning
alg = ['OPC', 'TPC', 'MPC', 'UC', 'QMO']
cross_f = [.1, .1, .1, .1, .4]
mut_e_f = [.9, .9, .9, .9, 0.]
mut_i_f = [.1, .1, .05, .1, .7]
mut_u_f = [0, 0, 0, .1, 0]


# Define initial population
def oggetti_da_inserire():
    return random.choices(range(0, 2), k=lunghezza_cromosoma)

# Define FITNESS function: "individuo" is one candidate i.e. an ideal backpack
def evaluate(individuo): 
    individuo = individuo[0]
    peso_totale = sum(x * y for x, y in zip(oggetti['Peso'], individuo))
    valore_totale = sum(x * y for x, y in zip(oggetti['Valore'], individuo))

    residuo = peso_totale - capienza
    if residuo > 0:
        residuo = 100

    # returns the residual space and the value of the candidate
    return abs(residuo), -1. * valore_totale

# CHROMOSOMS definition
def CreazionePopolazione():
    toolbox.register('oggetti_da_inserire', function=oggetti_da_inserire)
    # individuals and population definition
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.oggetti_da_inserire, n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # create population
    pop = toolbox.population(n=n_popolazione)
    return pop


# In this section, OPERATORS are defined

def MultiParentCrossover(pop, prob_cross):
    genitori = []
    offspring = []

    # SPLIT POPULATION IN PARENTS (WHO REPRODUCE) AND NON REPRODUCTIVE INDIVIDUALS 
    for i in range(n_popolazione):
        if random.random() < prob_cross:
            genitori.append(pop[i][0])
        else:
            offspring.append(pop[i][0])


    while len(offspring) < n_popolazione:

        mask = []
        for i in range(lunghezza_cromosoma):
            mask.append(random.randint(0, len(genitori) - 1))
        figlio = []
        for i in range(len(mask)):
            figlio.append(genitori[mask[i]][i])
        offspring.append(figlio)

    return offspring

# QUANTUM OPERATOR
def QMO(pop, prob_cross, prob_mut_i):

    genitori = []
    offspring = []

    # CHECK IF AN INDIVIDUAL WILL REPRODUCE
    for i in range(n_popolazione):
        if random.random() < prob_cross:
            genitori.append(pop[i][0])
        else:
            offspring.append(pop[i][0])

    # CREATE A MAPPING BY COUNTING THE FREQUENCIES OF 1 IN PARENTS
    mapping = []
    for j in range(lunghezza_cromosoma):
        temp = 0
        for i in genitori:
            temp += i[j]
        mapping.append(temp / len(genitori))

    # WITH THE PREVIOUS FREQUENCY COMPUTE ROTATION ANGLES
    theta = [np.pi * x for x in mapping]

    # CREATE QUANTUM CIRCUITS
    while len(offspring) < len(pop):
        qc = QuantumCircuit(lunghezza_cromosoma)

        # ROTATE ALONG Y
        for i in range(len(theta)):
            qc.ry(theta=theta[i], qubit=i)

        qc.barrier()
        # CASUAL MUTATION
        for i in range(len(theta)):
            if random.random() < prob_mut_i:
                angle = random.random() * np.pi
                qc.ry(theta=angle, qubit=i)

        
        qc.measure_all()

        # SINGLE SHOT SIMULATION
        sim = Aer.get_backend('qasm_simulator')
        result = sim.run(qc, shots=1, memory=True).result()

        # NOISE SIMULATION
        """
        device_backend = FakeGuadalupe()
        sim = AerSimulator.from_backend(device_backend)
        tcirc = transpile(qc, sim)
        result = sim.run(tcirc, shots=1, memory=True).result()
        """

        # TAKE CIRCUIT OUPTUT AND BUILD AN INDIVIDUAL
        memory = list(result.get_memory()[0])
        figlio = [int(x) for x in memory]

        # ADD IT TO OFFSPRING 
        offspring.append(figlio)

    return offspring

#CHECK THAT INDIVIDUALS CAN FIT IN THE BACKPACK 
def CheckBounds(capienza):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                # SE IL PESO DELL'INDIVIDUO E' MAGGIORE DELLA CAPIENZA DELLO ZAINO ALLORA MODIFICO I GENI
                if sum(child * oggetti['Peso']) > capienza:
                    # ESSENDO I GENI ORDINATI PER VALORE CRESCENTE, SWITCHO PROGRESSIVAMENTE GLI 1 FINCHE' NON RAGGIUNGO UN PESO INFERIORE ALLA CAPIENZA
                    for i in range(len(child)):
                        if child[i] == 1:
                            child[i] = 0
                        if sum(child * oggetti['Peso']) < capienza:
                            break
            return offspring

        return wrapper

    return decorator


#ANALYITIC APPROACH
def AnaliticAlgorithm():
  
    peso_totale = 0
    valore_totale = 0
    individuo = []

    for j in range(lunghezza_cromosoma):
        peso_totale += oggetti['Peso'][j]
        if peso_totale < capienza:
            individuo.append(1)
            valore_totale += oggetti['Valore'][j]
        else:
            peso_totale -= oggetti['Peso'][j]
            individuo.append(0)
    oggetti['Individuo'] = individuo

    return oggetti

#USE THE DIFFERENT ALGORITHMS
def Algorithm(pop, operator, prob_cross, prob_mut_i, prob_mut_e, prob_mut_u):

    toolbox.register('evaluate', evaluate)

    if operator == 'OPC': #ONE POINT CROSSOVER
        toolbox.register('one', tools.cxOnePoint)
        toolbox.decorate('one', CheckBounds(capienza=750))

    elif operator == 'TPC': #TWO POINT CROSSOVER
        toolbox.register('two', tools.cxTwoPoint)
        toolbox.decorate('two', CheckBounds(capienza=750))

    elif operator == 'UC': #UNIFORM CROSSOVER
        toolbox.register('uni', tools.cxUniform)
        toolbox.decorate('uni', CheckBounds(capienza=750))

    elif operator == 'MPC': #MULTI-PARENT CROSSOVER
        toolbox.register('multi', MultiParentCrossover, prob_cross=prob_cross)
        toolbox.decorate('multi', CheckBounds(capienza=750))

    elif operator == 'QMO': #QUANTISTIC MULTIPARENT OPERATOR
        toolbox.register('QMO', QMO, prob_cross=prob_cross, prob_mut_i=prob_mut_i)
        toolbox.decorate('QMO', CheckBounds(capienza=750))

    if operator != 'QMO': #questo perché in caso quantistico non bisogna mutare
        toolbox.register('mutate', tools.mutFlipBit, indpb=prob_mut_i)
        toolbox.decorate('mutate', CheckBounds(capienza=750))

    #PICK THE BEST INDIVIDUAL AMONG THE SELECTED ONE AND REPEAT K=3 TIMES
    toolbox.register('select', tools.selTournament, tournsize=3)

    g = 0
    migliori_individui = []

    # EVALUATE NEW INDIVIDUALS
    invalide_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalide_ind)

    for ind, fit in zip(invalide_ind, fitnesses):
        ind.fitness.values = evaluate(ind)

    # SELECT BEST ONE
    miglior_individuo = tools.selBest(pop, k=1, fit_attr='fitness')[0]
    migliori_individui.append(miglior_individuo[0])


    while g < n_generazioni:
        g += 1

        # POPULATION SELECTION, WORK ON CLONED ONES

        offspring = toolbox.select(pop, k=len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # CROSSOVER OPERATORS APPLICATION
        if operator == 'UC':
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < prob_cross:
                    toolbox.uni(child1[0], child2[0], indpb=prob_mut_u)
                    del child1.fitness.values
                    del child2.fitness.values

        elif operator == 'OPC':
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < prob_cross:
                    toolbox.one(child1[0], child2[0])
                    del child1.fitness.values
                    del child2.fitness.values


        elif operator == 'TPC':
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < prob_cross:
                    toolbox.two(child1[0], child2[0])
                    del child1.fitness.values
                    del child2.fitness.values

        elif operator == 'MPC':
            offspring = toolbox.multi(offspring)
            offspring = [creator.Individual([x]) for x in offspring]
            for ind in offspring:
                del ind.fitness.values


        elif operator == 'QMO':
            offspring = toolbox.QMO(offspring)
            offspring = [creator.Individual([x]) for x in offspring]
            for ind in offspring:
                del ind.fitness.values

        # APPLY MUTATION
        if operator != 'QMO':
            for mutant in offspring:
                if random.random() < prob_mut_e:
                    toolbox.mutate(mutant[0])
                    del mutant.fitness.values

        # EVALUATE INDIVUDUALS
        invalide_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalide_ind)

        for ind, fit in zip(invalide_ind, fitnesses):
            ind.fitness.values = evaluate(ind)

        # UPDATE POPULATION
        pop[:] = offspring

        # SELECT BEST ONE
        miglior_individuo = tools.selBest(pop, k=1, fit_attr='fitness')[0]
        migliori_individui.append(miglior_individuo[0])

    return migliori_individui

# RETURNS A LIST OF INDIVIDUALS TO EVALUATE
def Caratteristica(lista_individui, parameter):
    return [sum(i * oggetti[parameter]) for i in lista_individui]

# COMPUTE AVERAGE ON SEEDS
def Media(lista_di_liste):
    results = []
    for j in range(len(lista_di_liste[0])):
        temp = 0
        for i in range(len(lista_di_liste)):
            temp += lista_di_liste[i][j]
        results.append(temp / len(lista_di_liste))
    return results

# THIS IS ONE RUN FOR ONE SEED, NEEDS TO BE REPEATED
def Run(algoritmo, seme, prob_cross, prob_mut_i, prob_mut_e, prob_mut_u):
    random.seed(seme)
    print(f'seme: {seme}/{numero_semi}')
    popolazione = CreazionePopolazione()
    
    return Algorithm(operator=algoritmo, pop=popolazione, prob_cross=prob_cross, prob_mut_i=prob_mut_i,
                     prob_mut_e=prob_mut_e, prob_mut_u=prob_mut_u)
    
#PLOT 
def Grafico(nome, label, valori_medi, variabile):
    plt.xlabel('Generazioni')
    plt.ylabel(variabile)
    plt.plot(valori_medi, label=label)
    if algoritmo == 'QMO':
        plt.legend(title="Probabilita' Mutazione")
    else:
        plt.legend(title="Probabilita' Mutazione Interna")
    plt.savefig(f'Grafici/{nome}.png')   #HERE THE PLOTS ARE STORED, NEED TO CHANGE THIS OR MKDIR

#MAIN 
def Main(algorithm, numero_semi, prob_cross, prob_mut_i, prob_mut_e, prob_mut_u):
    valori_medi = []
    for seme in range(0, numero_semi):
        risultato = Run(algoritmo=algorithm, seme=seme, prob_cross=prob_cross, prob_mut_i=prob_mut_i,
                        prob_mut_e=prob_mut_e, prob_mut_u=prob_mut_u)
        valori_medi.append(Caratteristica(lista_individui=risultato, parameter='Valore'))
    return valori_medi


toolbox = base.Toolbox()

creator.create('FitnessMin', base.Fitness, weights=(-1., -1.))
creator.create('Individual', list, fitness=creator.FitnessMin)


plt.figure(figsize=(12, 9))
plt.axhline(y=1458, color='red', linestyle='--', alpha=.4)

#PLOT AND PRINT RESULTS
for index in range(len(alg)):
    algoritmo = alg[index]
    startTime = time.time()
    valori = Main(algorithm=algoritmo, numero_semi=numero_semi, prob_cross=cross_f[index], prob_mut_i=mut_i_f[index],
                  prob_mut_e=mut_e_f[index], prob_mut_u=mut_u_f[index])
    executionTime = round(time.time() - startTime, 2)
    print(f'Tempo di esecuzione:\t{executionTime} secondi')
    valori_zaino = Media(valori)
    Grafico(f"Valore Zaino con rumore", label=f'{algoritmo}', valori_medi=valori_zaino, variabile='Valore zaino')
    print(f'Il risultato ottenuto con {algoritmo} = {max(valori_zaino)}')
plt.close()



