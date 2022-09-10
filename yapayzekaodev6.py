import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

def soru1():
    bulasik_miktari = ctrl.Antecedent(np.arange(0, 101,15),"bulasik miktari")
    kirlilik_derecesi = ctrl.Antecedent(np.arange(0,101,15),"kirlilik derecesi")
    bulasik_cinsi = ctrl.Antecedent(np.arange(0,101,15),"bulasik cinsi")
    yikama_zamani = ctrl.Consequent(np.arange(30,161,10),"yikama zamani")
    deterjan_miktari = ctrl.Consequent(np.arange(0,93.5,17.5),"deterjan miktari")
    su_sicakligi = ctrl.Consequent(np.arange(35,68.5,15),"su sicakligi")
    ust_sepet = ctrl.Consequent(np.arange(2100,3501,200),"ust sepet pompa devri")
    alt_sepet = ctrl.Consequent(np.arange(2100,3501,200),"alt sepet pompa devri")

    bulasik_miktari.automf(3)
    kirlilik_derecesi.automf(3)
    bulasik_cinsi.automf(3)

    yikama_zamani["çok kısa"] = fuzz.trimf(yikama_zamani.universe,[30,30,60,])
    yikama_zamani["kısa"] = fuzz.trimf(yikama_zamani.universe, [40,60,90])
    yikama_zamani["orta"] = fuzz.trimf(yikama_zamani.universe,[70,100,120])
    yikama_zamani["uzun"] = fuzz.trimf(yikama_zamani.universe,[100,130,150])
    yikama_zamani["çok uzun"] = fuzz.trimf(yikama_zamani.universe,[130,150,160])
    deterjan_miktari["çok az"] = fuzz.trimf(deterjan_miktari.universe,[0,0,17.5])
    deterjan_miktari["az"] = fuzz.trimf(deterjan_miktari.universe,[17.5,32.5,42.5])
    deterjan_miktari["normal"] = fuzz.trimf(deterjan_miktari.universe,[32.5,57.5,67.5])
    deterjan_miktari["çok"] = fuzz.trimf(deterjan_miktari.universe,[57.5,82.5,92.5])
    deterjan_miktari["çok fazla"] = fuzz.trimf(deterjan_miktari.universe,[82.5,82.5,92.5])
    su_sicakligi["düşük"] = fuzz.trimf(su_sicakligi.universe,[35,37.5,50])
    su_sicakligi["normal"] = fuzz.trimf(su_sicakligi.universe,[37.5,50,67.5])
    su_sicakligi["yüksek"] = fuzz.trimf(su_sicakligi.universe,[55,55,67.5])
    ust_sepet["çok düşük"] = fuzz.trimf(ust_sepet.universe,[2100,2300,2400])
    ust_sepet["düşük"] = fuzz.trimf(ust_sepet.universe,[2300,2400,2700])
    ust_sepet["normal"] = fuzz.trimf(ust_sepet.universe,[2600,2700,3000])
    ust_sepet["yüksek"] = fuzz.trimf(ust_sepet.universe,[2900,3000,3300])
    ust_sepet["çok yüksek"] = fuzz.trimf(ust_sepet.universe,[3200,3300,3500])
    alt_sepet["çok düşük"] = fuzz.trimf(alt_sepet.universe,[2100,2300,2400])
    alt_sepet["düşük"] = fuzz.trimf(alt_sepet.universe,[2300,2400,2700])
    alt_sepet["normal"] = fuzz.trimf(alt_sepet.universe,[2600,2700,3000])
    alt_sepet["yüksek"] = fuzz.trimf(alt_sepet.universe,[2900,3000,3300])
    alt_sepet["çok yüksek"] = fuzz.trimf(alt_sepet.universe,[3200,3300,3500])

    bulasik_miktari.view()

    kirlilik_derecesi.view()

    bulasik_cinsi.view()

    yikama_zamani.view()

    deterjan_miktari.view()

    su_sicakligi.view()

    ust_sepet.view()

    alt_sepet.view()


    kural1 = ctrl.Rule(bulasik_miktari["poor"]|kirlilik_derecesi["poor"]|bulasik_cinsi["poor"],  ust_sepet["çok düşük"])
    kural2 = ctrl.Rule(bulasik_miktari["poor"]|kirlilik_derecesi["good"]|bulasik_cinsi["average"],ust_sepet["düşük"])
    kural3 = ctrl.Rule(bulasik_miktari["average"]|kirlilik_derecesi["average"]|bulasik_cinsi["good"], ust_sepet["yüksek"])
    kural4 = ctrl.Rule(bulasik_miktari["good"]|kirlilik_derecesi["good"]|bulasik_cinsi["average"],ust_sepet["düşük"])
    kural5 = ctrl.Rule(bulasik_miktari["poor"]|kirlilik_derecesi["good"]|bulasik_cinsi["average"],yikama_zamani["orta"])
    kural6 = ctrl.Rule(bulasik_miktari["average"]|kirlilik_derecesi["average"]|bulasik_cinsi["good"],yikama_zamani["orta"])
    kural7 = ctrl.Rule(bulasik_miktari["poor"]|kirlilik_derecesi["poor"]|bulasik_cinsi["poor"], yikama_zamani["kısa"])
    kural8= ctrl.Rule(bulasik_miktari["good"]|kirlilik_derecesi["good"]|bulasik_cinsi["average"],yikama_zamani["çok uzun"])
    kural9 = ctrl.Rule(bulasik_miktari["poor"]|kirlilik_derecesi["poor"]|bulasik_cinsi["poor"],deterjan_miktari["çok az"])            
    kural10 = ctrl.Rule(bulasik_miktari["poor"]|kirlilik_derecesi["good"]|bulasik_cinsi["average"],deterjan_miktari["normal"])
    kural11 = ctrl.Rule(bulasik_miktari["average"]|kirlilik_derecesi["average"]|bulasik_cinsi["good"],deterjan_miktari["normal"])                          
    kural12 = ctrl.Rule(bulasik_miktari["good"]|kirlilik_derecesi["good"]|bulasik_cinsi["average"],deterjan_miktari["çok fazla"])                  
    kural13 = ctrl.Rule(bulasik_miktari["poor"]|kirlilik_derecesi["poor"]|bulasik_cinsi["poor"],su_sicakligi["düşük"])
    kural14 = ctrl.Rule(bulasik_miktari["poor"]|kirlilik_derecesi["good"]|bulasik_cinsi["average"],su_sicakligi["yüksek"])                     
    kural15 = ctrl.Rule(bulasik_miktari["average"]|kirlilik_derecesi["average"]|bulasik_cinsi["good"],su_sicakligi["normal"])                  
    kural16 = ctrl.Rule(bulasik_miktari["good"]|kirlilik_derecesi["good"]|bulasik_cinsi["average"],su_sicakligi["yüksek"])         
    kural17 = ctrl.Rule(bulasik_miktari["poor"]|kirlilik_derecesi["poor"]|bulasik_cinsi["poor"],alt_sepet["çok düşük"])
    kural18 = ctrl.Rule(bulasik_miktari["poor"]|kirlilik_derecesi["good"]|bulasik_cinsi["average"],alt_sepet["çok yüksek"])
    kural19 = ctrl.Rule(bulasik_miktari["poor"]|kirlilik_derecesi["good"]|bulasik_cinsi["average"],alt_sepet["yüksek"])
    kural20 = ctrl.Rule(bulasik_miktari["good"]|kirlilik_derecesi["good"]|bulasik_cinsi["average"],alt_sepet["çok yüksek"])
        
    yikama_zamaniKontrol = ctrl.ControlSystem([kural5,kural6,kural7,kural8])
    yikama_zamaniBelirleme = ctrl.ControlSystemSimulation(yikama_zamaniKontrol)
    deterjan_miktariKontrol = ctrl.ControlSystem([kural9,kural10,kural11,kural12])
    deterjan_miktariBelirleme = ctrl.ControlSystemSimulation(deterjan_miktariKontrol)
    su_sicakligiKontrol = ctrl.ControlSystem([kural13,kural14,kural15,kural16])
    su_sicakligiBelirleme = ctrl.ControlSystemSimulation(su_sicakligiKontrol)
    ust_sepetKontrol = ctrl.ControlSystem([kural1,kural2,kural3,kural4])
    ust_sepetBelirleme = ctrl.ControlSystemSimulation(ust_sepetKontrol)
    alt_sepetKontrol = ctrl.ControlSystem([kural17,kural18,kural19,kural20])
    alt_sepetBelirleme = ctrl.ControlSystemSimulation(alt_sepetKontrol)

    yikama_zamaniBelirleme.input["bulasik miktari"] = 50
    yikama_zamaniBelirleme.input["kirlilik derecesi"] = 10
    yikama_zamaniBelirleme.input["bulasik cinsi"] = 10
    yikama_zamaniBelirleme.compute()
    print(yikama_zamaniBelirleme.output["yikama zamani"])

    deterjan_miktariBelirleme.input["bulasik miktari"] = 50
    deterjan_miktariBelirleme.input["kirlilik derecesi"] = 50
    deterjan_miktariBelirleme.input["bulasik cinsi"] = 50
    deterjan_miktariBelirleme.compute()
    print(deterjan_miktariBelirleme.output["deterjan miktari"])

    su_sicakligiBelirleme.input["bulasik miktari"] = 50
    su_sicakligiBelirleme.input["kirlilik derecesi"] = 50
    su_sicakligiBelirleme.input["bulasik cinsi"] = 50
    su_sicakligiBelirleme.compute()
    print(su_sicakligiBelirleme.output["su sicakligi"])

    ust_sepetBelirleme.input["bulasik miktari"] = 50
    ust_sepetBelirleme.input["kirlilik derecesi"] = 50
    ust_sepetBelirleme.input["bulasik cinsi"] = 50
    ust_sepetBelirleme.compute()
    print(ust_sepetBelirleme.output["ust sepet pompa devri"])

    alt_sepetBelirleme.input["bulasik miktari"] = 50
    alt_sepetBelirleme.input["kirlilik derecesi"] = 50
    alt_sepetBelirleme.input["bulasik cinsi"] = 50
    alt_sepetBelirleme.compute()
    print(alt_sepetBelirleme.output["alt sepet pompa devri"])

    yikama_zamani.view(sim=yikama_zamaniBelirleme)
    deterjan_miktari.view(sim=deterjan_miktariBelirleme)
    su_sicakligi.view(sim=su_sicakligiBelirleme)
    ust_sepet.view(sim=ust_sepetBelirleme)
    alt_sepet.view(sim=alt_sepetBelirleme)


soru1()