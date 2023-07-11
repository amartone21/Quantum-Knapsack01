import random
from qiskit import *
import numpy as np
from deap import tools, base, creator
import pandas as pd
import matplotlib.pyplot as plt
import time
from qiskit.providers.aer import AerSimulator
from qiskit.test.mock import FakeGuadalupe

'''Script che cerca l'indivuo migliore, inteso come combinazione di oggetti presenti nello zaino rispettando il vincolo del peso massimo di 750 unità. 
Il focus è sulla comparazione tra algoritmo analitico, genetico e quantistico.
L'invididuo migliore ha e ha come valore 1458 e peso 749. (dati dal creatore del dataset: https://people.sc.fsu.edu/~jburkardt/datasets/knapsack_01/knapsack_01.html 
voce P07.
Per sperimentare e provare cose nuove, oltre alle solite librerie Pandas e numpy, ho voluto provare qiskit e deap.
Riporto in seguito le tecniche usate ed i rispettivi score, dai quali si evince che l'utilizzo della tecnologia quantistica non apporta alcun vantaggio

•One-point = 1448,2
• Uniform = 1446,9
• Multiparent = 1444,8
• Two-Point = 1444,0
• QMO = 1441,3
• Analitico = 1441

La cifra decimale deriva dalla media sui seed ed è soggetta ad errore statistico

NOTA: Lo script salva il grafico finale in una cartella chiamata Grafici che era già presente sul mio dispositivo, 
avrei voluto implementare un semplice mkdir ma non sapevo se per la correzione, girando sul Suo pc, la cosa potrebbe essere inappropriata.
In ogni caso credo che senza la cartella ci sia un problema per salvare il grafico.


'''

# DEFINIZIONE DEL DATASET
#   NOME    PESO    VALORE
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

#definizione dei vincoli e dei parametri
capienza = 750
n_popolazione = 30

n_generazioni = 30
numero_semi = 50

#valori ottimali ottenuti da tuning
alg = ['OPC', 'TPC', 'MPC', 'UC', 'QMO']
cross_f = [.1, .1, .1, .1, .4]
mut_e_f = [.9, .9, .9, .9, 0.]
mut_i_f = [.1, .1, .05, .1, .7]
mut_u_f = [0, 0, 0, .1, 0]


# DEFINISCO LA POPOLAZIONE INIZIALE
def oggetti_da_inserire():
    return random.choices(range(0, 2), k=lunghezza_cromosoma)

# DEFINISCO LA FUNZIONE DI FITNESS
def evaluate(individuo): #individuo è un candidato zaino ideale
    individuo = individuo[0]
    peso_totale = sum(x * y for x, y in zip(oggetti['Peso'], individuo))
    valore_totale = sum(x * y for x, y in zip(oggetti['Valore'], individuo))

    residuo = peso_totale - capienza
    if residuo > 0:
        residuo = 100

    # RESTITUISCE LO SPAZIO RESIDUO O SUPERFLUO E IL VALORE TOTALE DELL'INDIVIDUO
    return abs(residuo), -1. * valore_totale

# DEFINISCO I CROMOSOMI
def CreazionePopolazione():
    toolbox.register('oggetti_da_inserire', function=oggetti_da_inserire)
    # DEFINISCO I SINGOLI INDIVIDUI E LA POPOLAZIONE COMPOSTA DA QUESTI
    toolbox.register('individual', tools.initRepeat, creator.Individual, toolbox.oggetti_da_inserire, n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    # CREO LA POPOLAZIONE
    pop = toolbox.population(n=n_popolazione)
    return pop


# OPERATORI

# OPERATORE GENETICO DI CROSSOVER MULTIPARENT
def MultiParentCrossover(pop, prob_cross):
    genitori = []
    offspring = []

    # SI DIVIDE LA POPOLAZIONE NELLA PARTE CHE SI RIPRODUCE (genitori) E NELLA PARTE CHE SI PROPAGA SENZA RIPRODURSI
    for i in range(n_popolazione):
        if random.random() < prob_cross:
            genitori.append(pop[i][0])
        else:
            offspring.append(pop[i][0])

    # FINCHE' GLI INDIVIDUI IN offspring SONO MINORI DELLA POPOLAZIONE FISSATA...
    while len(offspring) < n_popolazione:

        # GENERO UNA LISTA DI lunghezza_cromosoma DI LUNGHEZZA I CUI VALORI SONO COMPRESI TRA 0 E IL NUMERO DI GENITORI
        # INDICANTE, GENE PER GENE, DA QUALE GENITORE PRENDERE IL VALORE
        mask = []
        for i in range(lunghezza_cromosoma):
            mask.append(random.randint(0, len(genitori) - 1))
        figlio = []
        for i in range(len(mask)):
            figlio.append(genitori[mask[i]][i])
        offspring.append(figlio)

    return offspring

# OPERATORE QUANTISTICO
def QMO(pop, prob_cross, prob_mut_i):

    genitori = []
    offspring = []

    # PER OGNI INDIVIDUO VEDO SE FARA' PARTE DELLA POPOLAZIONE CHE SI RIPRODURRA'
    for i in range(n_popolazione):
        if random.random() < prob_cross:
            genitori.append(pop[i][0])
        else:
            offspring.append(pop[i][0])

    # FACCIO UN MAPPING MISURANDO LA FREQUENZA DI 1 NELL'INSIEME DEI GENITORI
    mapping = []
    for j in range(lunghezza_cromosoma):
        temp = 0
        for i in genitori:
            temp += i[j]
        mapping.append(temp / len(genitori))

    # CALCOLO GLI ANGOLI DI ROTAZIONE IN FUNZIONE DELLA FREQUENZA CALCOLATA PRECEDENTEMENTE
    theta = [np.pi * x for x in mapping]

    # QUI VADO A CREARE I MIEI CIRCUITI QUANTISTICI PER POTER GENERARE I FIGLI
    while len(offspring) < len(pop):
        qc = QuantumCircuit(lunghezza_cromosoma)

        # FACCIO UNA ROTAZIONE SU Y IN BASE ALLA FREQUENZA PER OGNI QUBIT
        for i in range(len(theta)):
            qc.ry(theta=theta[i], qubit=i)

        qc.barrier()
        # FACCIO AVVENIRE LA MUTAZIONE IN MANIERA CASUALE SU OGNI QUBIT
        for i in range(len(theta)):
            if random.random() < prob_mut_i:
                angle = random.random() * np.pi
                qc.ry(theta=angle, qubit=i)

        # MISURO IL CIRCUITO
        qc.measure_all()

        # APRO IL SIMULATORE E LO FACCIO PARTIRE PER UN SINGOLO SHOT
        sim = Aer.get_backend('qasm_simulator')
        result = sim.run(qc, shots=1, memory=True).result()

        # SIMULATORE DI RUMORE
        """
        device_backend = FakeGuadalupe()
        sim = AerSimulator.from_backend(device_backend)
        tcirc = transpile(qc, sim)
        result = sim.run(tcirc, shots=1, memory=True).result()
        """

        # PRENDO IL VALORE CHE E' STATO FORNITO DAL CIRCUITO E LO TRASFORMO IN UN INDIVIDUO
        memory = list(result.get_memory()[0])
        figlio = [int(x) for x in memory]

        # AGGIUNGO QUESTO INDIVIDUO ALLA PROLE
        offspring.append(figlio)

    return offspring

#CONTROLLO CHE GLI INDIVIDUI COSTRUITI POSSANO EFFETTIVAMENTE STARE NELLO ZAINO 
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


# CONFRONTIAMO GLI ALGORITMI GENETICI, CLASSICI E QUANTISTICO, CON UN APPROCCIO ANALITICO:
def AnaliticAlgorithm():
    # INIZIALIZZO LE VARIABILI
    peso_totale = 0
    valore_totale = 0
    individuo = []

    # SI VALUTA IL RAPPORTO VALORE/PESO E SI ORDINANO GLI OGGETTI DI CONSEGUENZA
    # IL MIGLIOR INDIVIDUO VIENE SELEZIONATO AGGIUNGENDO PROGRESSIVAMENTE GLI OGGETTI CON IL RAPPORTO MAGGIORE
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

#UTILIZZO I DIVERSI ALGORITMI
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

    # SELEZIONO IL MIGLIOR INDIVIDUO TRA I tournsize INDIVIDUI SCELTI CASUALMENTE E LO FACCIO k VOLTE
    toolbox.register('select', tools.selTournament, tournsize=3)

    g = 0
    migliori_individui = []

    # VALUTAZIONE DEI NUOVI INDIVIDUI
    invalide_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalide_ind)

    for ind, fit in zip(invalide_ind, fitnesses):
        ind.fitness.values = evaluate(ind)

    # SELEZIONO IL MIGLIOR INDIVIDUO
    miglior_individuo = tools.selBest(pop, k=1, fit_attr='fitness')[0]
    migliori_individui.append(miglior_individuo[0])


    while g < n_generazioni:
        g += 1

        # SELEZIONE DELLA POPOLAZIONE E CLONAZIONE E SI LAVORA SUGLI ELEMENTI CLONATI

        offspring = toolbox.select(pop, k=len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # APPLICAZIONE DEGLI OPERATORI DI CROSSOVER
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

        # APPLICO L'OPERATORE DI MUTAZIONE SOLTANTO PER GLI OPERATORI DI CROSSOVER CLASSICI
        if operator != 'QMO':
            for mutant in offspring:
                if random.random() < prob_mut_e:
                    toolbox.mutate(mutant[0])
                    del mutant.fitness.values

        # VALUTAZIONE DEI NUOVI INDIVIDUI
        invalide_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalide_ind)

        for ind, fit in zip(invalide_ind, fitnesses):
            ind.fitness.values = evaluate(ind)

        # AGGIORNO LA POPOLAZIONE CON I NUOVI INDIVIDUI APPENA OTTENUTI
        pop[:] = offspring

        # SELEZIONO IL MIGLIOR INDIVIDUO
        miglior_individuo = tools.selBest(pop, k=1, fit_attr='fitness')[0]
        migliori_individui.append(miglior_individuo[0])

    return migliori_individui

# RESTITUISCE UNA LISTA CON IL VALORE DI TUTTI GLI INDIVIDUI
# USATA PER CALCOLARE LA MEDIA SUI SEMI
def Caratteristica(lista_individui, parameter):
    return [sum(i * oggetti[parameter]) for i in lista_individui]

# L'EFFETTIVO CALCOLO DELLA MEDIA SUI SEMI
def Media(lista_di_liste):
    results = []
    for j in range(len(lista_di_liste[0])):
        temp = 0
        for i in range(len(lista_di_liste)):
            temp += lista_di_liste[i][j]
        results.append(temp / len(lista_di_liste))
    return results

# SINGOLA RUN PER SINGOLO SEME
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
    plt.savefig(f'Grafici/{nome}.png')   #Qui è dove salva i grafici e c'è il problema della cartella

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

#STAMPA A SCHERMO E GRAFICO I VALORI OTTENUTI 
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



