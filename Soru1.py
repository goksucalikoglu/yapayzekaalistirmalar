from random import choice
from experta import *

class diş_ağrısı(Fact):
    pass

class dişhekiminegitme(KnowledgeEngine):
    @Rule(diş_ağrısı(ağrı="diş fırçalarken diş eti kanaması"))
    def diş_eti_kanaması(self):    
        print("Diş fırçalarken diş eti kanaması: Diş hastalığı vardır. Diş hekimine başvurunuz.")
    
    @Rule(diş_ağrısı(ağrı="diş fırçalarken uzun süreli diş eti kanaması"))
    def uzun_süreli_diş_eti_kanaması(self):
        print("Diş fırçalarken uzun süreli diş eti kanaması: Diş eti çekilmesi vardır. Diş hekimine başvurunuz.")
   
    @Rule(diş_ağrısı(ağrı="diş eti çekilmesi var ve diş kökü görünüyor"))
    def diş_eti_çekilmesi_var_ve_diş_kökü_görünüyor(self):
        print("Diş eti çekilmesi var ve diş kökü görünüyor: Dolgu yaptırınız.")
    
    @Rule(diş_ağrısı(ağrı="dişte yiyecek ve içeceklerden oluşan renk değişimi var")) 
    def diş_renk_değişimi(self):
        print("Dişte yiyecek ve içeceklerden oluşan renk değişimi var: Dişleri temizletiniz.")
    
    @Rule(diş_ağrısı(ağrı="yeni diş çıkarken morarma görünüyor"))
    def yeni_diş_çıkarken_morarma(self):
        print("Yeni diş çıkarken morarma görünüyor: Diş hekimine başvurunuz.")
    
    @Rule(diş_ağrısı(ağrı="dişte ağrı yapmayan çürük var"))
    def ağrı_yapmayan_çürük_var(self):
        print("Dişte ağrı yapmayan çürük var: Dolgu yaptırınız.")
    
    @Rule(diş_ağrısı(ağrı="dişteki çürük ileri derece"))
    def çürük_ileri_derece(self):
        print("Dişteki çürük ileri derece: Kanal tedavisi ve dolgu yaptırınız.")
        
uzman = dişhekiminegitme()
uzman.reset()
uzman.declare(diş_ağrısı(ağrı=choice(["diş fırçalarken diş eti kanaması",
                                      "diş fırçalarken uzun süreli diş eti kanaması",
                                      "diş eti çekilmesi var ve diş kökü görünüyor",
                                      "dişte yiyecek ve içeceklerden oluşan renk değişimi var",
                                      "yeni diş çıkarken morarma görünüyor",
                                      "dişte ağrı yapmayan çürük var",
                                      "dişteki çürük ileri derece"])))
uzman.run()
        
        