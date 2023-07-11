import random
from qiskit import *
import numpy as np
from deap import tools, base, creator
import pandas as pd
import matplotlib.pyplot as plt

'''
Algoritmo per il tuning dei parametri, attraverso il testing sequenziale di diverse combinazioni. 
Quello su cui si va ad agire sono le probabilità di crossover. 
Popolazione fissata a 30, generazioni fissate a 25 (valore di plateau nelle performance).
Valore dello zaino mediato su 30 esecuzioni partendo da 30 seed.

NOTA: Lo script salva il grafico finale in  cartelle ( e e sottocartelle ) chiamate Grafici/NOME_ALGORITMO che erano già presenti sul mio dispositivo, 
avrei voluto implementare un semplice mkdir ma non sapevo se per la correzione, girando sul Suo pc, la cosa potrebbe essere inappropriata, 
visto che si creerebbero 6 sottocartelle diverse, ognuna contenente molte immagini
In ogni caso credo che senza le cartelle ci sia un problema per salvare i grafici.



'''
#definizione parametri iniziali 
capienza = 750
n_popolazione = 30
n_generazioni = 25
numero_semi = 40


cross_p = [.3, .4, .5, .8, 1.]
mut_e_p = [.1, .3, .5, .7, .9, 1.]
mut_i_p = [.01, .05, .1, .3, .5, .7, .9]
mut_u_p = [.01, .05, .1, .2, .5, .8, .1]

algoritmo = 'UC'

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


# DEFINISCO LA POPOLAZIONE INIZIALE
def oggetti_da_inserire():
    return random.choices(range(0, 2), k=lunghezza_cromosoma)

# DEFINISCO LA FUNZIONE DI FITNESS
def evaluate(individuo):
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

# OPERATORE DI CROSSOVER MULTIPARENT
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

        # PRENDO IL VALORE CHE E' STATO FORNITO DAL CIRCUITO E LO TRASFORMO IN UN INDIVIDUO
        memory = list(result.get_memory()[0])
        figlio = [int(x) for x in memory]

        # AGGIUNGO QUESTO INDIVIDUO ALLA PROLE
        offspring.append(figlio)

    return offspring

#Check sulle dimensioni
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


def Algorithm(pop, operator, prob_cross, prob_mut_i, prob_mut_e, prob_mut_u):

    toolbox.register('evaluate', evaluate)

    if operator == 'OPC':
        toolbox.register('one', tools.cxOnePoint)
        toolbox.decorate('one', CheckBounds(capienza=750))

    elif operator == 'TPC':
        toolbox.register('two', tools.cxTwoPoint)
        toolbox.decorate('two', CheckBounds(capienza=750))

    elif operator == 'UC':
        toolbox.register('uni', tools.cxUniform)
        toolbox.decorate('uni', CheckBounds(capienza=750))

    elif operator == 'MPC':
        toolbox.register('multi', MultiParentCrossover, prob_cross=prob_cross)
        toolbox.decorate('multi', CheckBounds(capienza=750))

    elif operator == 'QMO':
        toolbox.register('QMO', QMO, prob_cross=prob_cross, prob_mut_i=prob_mut_i)
        toolbox.decorate('QMO', CheckBounds(capienza=750))

    if operator != 'QMO':
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

            print('offspring', offspring)
            print('::2', offspring[::2])
            print('1::2', offspring[1::2])


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

def Grafico(nome, label, valori_medi, variabile):
    plt.xlabel('Generazioni')
    plt.ylabel(variabile)
    plt.plot(valori_medi, label=label)
    if algoritmo == 'QMO':
        plt.legend(title="Probabilita' Mutazione")
    else:
        plt.legend(title="Probabilita' Mutazione Interna")
    plt.savefig(f'Grafici/{nome}.png')


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


valore_massimo = 0
tuning = ''

if algoritmo == 'UC':
    for cross in cross_p:
        for mut_e in mut_e_p:
            for mut_u in mut_u_p:
                plt.figure(figsize=(12, 9))
                plt.axhline(y=1458, color='red', linestyle='--', alpha=.4)
                for mut_i in mut_i_p:
                    # IN algorithm METTO L'ALGORITMO DI CUI VOGLIO FARE IL TUNING
                    valori = Main(algorithm=algoritmo, numero_semi=numero_semi, prob_cross=cross, prob_mut_i=mut_i, prob_mut_e=mut_e, prob_mut_u=mut_u)
                    valori_zaino = Media(valori)
                    # PER IL TUNING VADO A FARE LA MEDIA SUL PLATEAU E SALVO I PARAMETRI CHE MI CONSENTONO DI ARRIVARE A UN VALORE MAGGIORE
                    if np.mean(valori_zaino[5:]) > valore_massimo:
                        valore_massimo = np.mean(valori_zaino)
                        tuning = f'{algoritmo}/cross:{round(cross, 3)}mut_e:\t{round(mut_e, 3)}\tmut_i:{round(mut_i, 3)}\tmut_u:{round(mut_u, 3)}'
                    # GRAFICO TUTTO
                    Grafico(f"{algoritmo}/cross {round(cross, 3)} mut_e {round(mut_e, 3)} mut_u {round(mut_u, 3)}", label=f'{round(mut_i, 3)}', valori_medi=valori_zaino, variabile='Valore zaino')
                plt.close()
    print(tuning)

elif algoritmo == 'QMO':
    for cross in cross_p:
        plt.figure(figsize=(12, 9))
        plt.axhline(y=1458, color='red', linestyle='--', alpha=.4)
        for mut_i in mut_i_p:
            # IN algorithm METTO L'ALGORITMO DI CUI VOGLIO FARE IL TUNING
            valori = Main(algorithm=algoritmo, numero_semi=numero_semi, prob_cross=cross, prob_mut_i=mut_i, prob_mut_e=0., prob_mut_u=0.)
            valori_zaino = Media(valori)
            # PER IL TUNING VADO A FARE LA MEDIA SUL PLATEAU E SALVO I PARAMETRI CHE MI CONSENTONO DI ARRIVARE A UN VALORE MAGGIORE
            if np.mean(valori_zaino[5:]) > valore_massimo:
                valore_massimo = np.mean(valori_zaino)
                tuning = f'{algoritmo}/cross:{round(cross, 3)}mut:\t{round(mut_i, 3)}'
            # GRAFICO TUTTO
            Grafico(f"{algoritmo}/cross {round(cross, 3)}", label=f'{round(mut_i, 3)}', valori_medi=valori_zaino, variabile='Valore zaino')
        plt.close()
    print(tuning)

elif algoritmo == 'OPC' or algoritmo == 'TPC' or algoritmo == 'MPC':
    for cross in cross_p:
        for mut_e in mut_e_p:
            plt.figure(figsize=(12, 9))
            plt.axhline(y=1458, color='red', linestyle='--', alpha=.4)
            for mut_i in mut_i_p:
                # IN algorithm METTO L'ALGORITMO DI CUI VOGLIO FARE IL TUNING
                valori = Main(algorithm=algoritmo, numero_semi=numero_semi, prob_cross=cross, prob_mut_i=mut_i, prob_mut_e=mut_e, prob_mut_u=0.)
                valori_zaino = Media(valori)
                # PER IL TUNING VADO A FARE LA MEDIA SUL PLATEAU E SALVO I PARAMETRI CHE MI CONSENTONO DI ARRIVARE A UN VALORE MAGGIORE
                if np.mean(valori_zaino[5:]) > valore_massimo:
                    valore_massimo = np.mean(valori_zaino)
                    tuning = f'{algoritmo}/cross:{round(cross, 3)}\tmut_e:{round(mut_e, 3)}\tmut_i:{round(mut_i, 3)}'
                # GRAFICO TUTTO
                Grafico(f"{algoritmo}/cross {round(cross, 3)} mut_e {round(mut_e, 3)}", label=f'{round(mut_i, 3)}', valori_medi=valori_zaino, variabile='Valore zaino')
            plt.close()
    print(tuning)




