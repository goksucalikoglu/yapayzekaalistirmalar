import numpy as np
import pandas as pd

def soru1():
    aAcısı=int(input("1. Açıyı Giriniz: "))
    bAcısı=int(input("2. Açıyı Giriniz: "))
    cAcısı=int(input("3. Açıyı Giriniz: "))

    if aAcısı+bAcısı+cAcısı>180: print("Bu Bir Üçgen Değildir.")
    elif aAcısı==90 or bAcısı==90 or cAcısı==90:print("Bu Üçgen 'Dik' Üçgendir.")
    elif aAcısı<90 and bAcısı<90 and cAcısı<90:print("Bu Üçgen 'Dar' Üçgendir.")
    elif aAcısı>90 or bAcısı>90 or cAcısı>90: print("Bu Üçgen 'Geniş' Üçgendir.")
    else:print("Bu Bir Üçgen Değildir.")


def soru2():
    uzayli_rengi=str(input("Uzaylı Rengi Giriniz: ")).str.lower();
    if uzayli_rengi=="yeşil":
        print("Tebrikler, yeşil uzaylıya ateş ettiğiniz için 5 puan kazandınız")
    else:
        print("Tebrikler, yeşil olmayan uzaylıya ateş ettiğiniz için 10 puan kazandınız")


def soru3():
    uzayli_rengi=str(input("Uzaylı Rengi Giriniz: ")).str.lower();
    if uzayli_rengi=="yeşil":
        print("Tebrikler, yeşil uzaylıya ateş ettiğiniz için 5 puan kazandınız")
    elif uzayli_rengi=="sarı":
        print("Tebrikler, sarı uzaylıya ateş ettiğiniz için 10 puan kazandınız")
    elif uzayli_rengi=="kırmızı":
        print("Tebrikler, kırmızı uzaylıya ateş ettiğiniz için 15 puan kazandınız")
    else:
        print("Malesef Herhangi bir uzaylıya ateş edemediniz -10 puan.")


def soru4():
    yas=int(input("Lütfen Yaş Giriniz: "));
    if yas<2: print("Bu kişi bebektir")
    elif yas>=2 and yas<4: print("Bu kişi yeni yürümeye başlayan çocuktur.")
    elif yas>=4 and yas<13: print("Bu kişi çocuktur")
    elif yas>=13 and yas<20: print("Bu kişi ergendir")
    elif yas>=20 and yas<65:print("Bu kişi yetişkindir")
    elif yas>=65:print("Bu kişi yaşlıdır")
    

def soru5():
    favori_meyveler=["elma", "armut","karpuz", "kavun", "muz"]
    ornek_meyveler="elma,armut,karpuz,kavun,muz,portakal,çilek,vişne,kiraz,mandalina"
    ornek_meyveler=ornek_meyveler.split(',')
    for i in ornek_meyveler:
         if i in favori_meyveler: print(i + " Favori")
         else: print(i + " Favori Değil")


