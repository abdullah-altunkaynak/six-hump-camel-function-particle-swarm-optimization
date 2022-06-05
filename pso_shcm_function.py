# -*- coding: utf-8 -*-
"""
Created on Sun Jul 05 11:35:31 2022

@author: Abdullah
"""
import random
import numpy as np

class PSO:
    def __init__(self, en_dusuk, en_yuksek, parcacik_sayisi, degisken_sayisi , c1, c2, rand1, rand2):
        self.en_dusuk = en_dusuk
        self.en_yuksek = en_yuksek
        self.parcacik_sayisi = parcacik_sayisi
        self.degisken_sayisi = degisken_sayisi
        self.c1 = c1
        self.c2 = c2
        self.rand1 = rand1
        self.rand2 = rand2
        self.parcaciklar = np.random.uniform(low= self.en_dusuk, high= self.en_yuksek, size=(self.parcacik_sayisi, self.degisken_sayisi))
        self.uygunluk = np.zeros(self.parcacik_sayisi)
        self.pbest = np.zeros((self.parcacik_sayisi, (self.degisken_sayisi) + 1))
        self.gbest = np.zeros((self.degisken_sayisi) + 1)
    def six_hump_camel_back(self, x1, x2):
        uygunluk = (4 * x1**2) - (2.1 * x1**4) + (1/3 * x1**6) + (x1 * x2) - (4 * x2**2) + (4 * x2**4)
        return uygunluk
    def hiz_fonksiyonu(self, x, rand1, c1, pbest, rand2, c2, gbest):
        hiz = 0.2 * x + (c1 * rand1 * (pbest - x)) + (c2 * rand2 * (gbest - x))
        return hiz
    def konum_guncelleme_fonksiyonu(self, x, rand1, c1, pbest, rand2, c2, gbest):
        x = x + self.hiz_fonksiyonu(x, rand1, c1, pbest, rand2, c2, gbest)
        return x
    def aralık_kontrol_fonksiyonu(self, x):
        if(x < self.en_dusuk):
            x = self.en_dusuk
        if(x > self.en_yuksek):
            x = self.en_yuksek
        return x
    def train(self):
        self.uygunluk = self.six_hump_camel_back(self.parcaciklar[:,0], self.parcaciklar[:,1])
        
        self.pbest[:,0:2] = self.parcaciklar[:,0:2] # ilk durumda pbestler kendisi
        self.pbest[:,2] = self.uygunluk #ilk durumda en iyi uygunlukta kendisinin ki
        self.gbest = self.pbest[self.pbest[:,2].argmin(),:] # uygunluk değeri en az olan satırı komple alıyoruz ve 2. index uygunluktur
        self.x = 0
        while(self.gbest[2] >= 0.0001):
            self.x += 1
            self.uygunluk = self.six_hump_camel_back(self.parcaciklar[:,0], self.parcaciklar[:,1]) # her iterasyonda uygunluk hesapla
            for i in range(self.uygunluk.size): #uygunluk kadar dönecek
                if(self.pbest[i,2] >= self.uygunluk[i]): # eğer şuanki uygunluk bir sonrakinden büyükse pbesti güncelle
                    self.pbest[i,2] = self.uygunluk[i]
                    self.pbest[i,1] = self.parcaciklar[i,1]
                    self.pbest[i,0] = self.parcaciklar[i,0]
            if(self.gbest[2] >= self.pbest[self.pbest[:,2].argmin(), 2]): # gbesti güncelliyoruz
                self.gbest = self.pbest[self.pbest[:,2].argmin(), :]
            self.parcaciklar[:,0] = self.konum_guncelleme_fonksiyonu(self.parcaciklar[:,0], self.rand1, self.c1, self.pbest[:,0], self.rand2, self.c2, self.gbest[0])
            self.parcaciklar[:,1] = self.konum_guncelleme_fonksiyonu(self.parcaciklar[:,1], self.rand1, self.c1, self.pbest[:,1], self.rand2, self.c2, self.gbest[1])
            for i in range(self.parcaciklar[:,0].size):
                self.parcaciklar[i,0] = self.aralık_kontrol_fonksiyonu(self.parcaciklar[i,0])
                self.parcaciklar[i,1] = self.aralık_kontrol_fonksiyonu(self.parcaciklar[i,1])
        return self.gbest;
psoNesnesi = PSO(-5, 5, 10, 2, 2, 2, 2, 2)
gbest_degerleri = psoNesnesi.train()
print("en iyi x1 değeri: ", psoNesnesi.gbest[0])
print("\nen iyi x2 değeri: ", psoNesnesi.gbest[1])
print("\nBulunan uygunluk değeri: ", psoNesnesi.gbest[2])
print("\nBu sonuç: ", psoNesnesi.x, " iterasyondan sonra bulunmuştur!")