def soru1():
    try:
        inputA=int(input("Lütfen 1 ile 10 arasında bir sayı giriniz: "))
        if(inputA>=1 and inputA<=10):
            print(str(inputA)+" İçin : ",end="")
            for x in range(1,11,1):
                print(str(x*inputA)+" ",end="")
            print("\n")
        else:
            print("Hatalı Giriş Yaptınız")
            soru1()
    except:
        print("Hatalı Giriş Yaptınız")
        soru1()


    


def soru2():
    inputA=input("Lütfen bir sayı giriniz: ")
    try:
        inputB = inputA.replace(".","").replace(",","").replace(" ","")
        inputC = float(inputB)
        i=0
        while(i<len(inputB)):
           i+=1
        print(str(inputA) + " Sayısı "+str(i)+" Basamaklıdır")
    except:
        print("Hatalı Giriş Yaptınız")
        soru2()


def soru3():
    sayisalDegerler = [12, 15, 32, 42, 55, 75, 122, 132, 150, 180, 200]
    print("For İçin: ",end="")
    for x in sayisalDegerler:
        if(x>150):
            continue
        elif(x%5==0):
            print(str(x)+" ",end="")
    print("\nWhile İçin: ",end="")
    i=0
    while(i<len(sayisalDegerler)):
        x=sayisalDegerler[i]
        i+=1
        if(x>150):
            continue
        elif(x%5==0):
            print(str(x)+" ",end="")    
            
    print("\n")


def soru4():
    try:
        inputA=int(input("A için Sayı Giriniz: "))
        inputB=int(input("B için Sayı Giriniz: "))
        inputC=int(input("C için Sayı Giriniz: "))
        i=0
        for x in range(inputA,inputB+1,1):
            if(x%inputC==0):
                i+=1
        print("C ye Bölünebilen Toplam "+ str(i) + " Adet Sayı Vardır")
    except:
        print("Hatalı Giriş Yaptınız")
        soru4()

def soru5():
    for x in range(1,100,1):
        print(str(x) + " - "+ str(100 - x))


def soru6():
    inputA=input("Lütfen IP Adresinizi Giriniz: ")
    inputA=list(map(int, inputA.split(".")))
    if(len(inputA)!=4):
        print("Hatalı Giriş Yaptınız")
        soru6()
    else:
        j=0
        while(j<50):
            i=3
            while(i>=0):
                part=inputA[i]
                if(part==255):
                    inputA[i]=0
                    i-=1
                else:
                    inputA[i]+=1
                    break
            j+=1
            print('.'.join(map(str, inputA)))


soru6()