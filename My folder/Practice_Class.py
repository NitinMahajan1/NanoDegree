# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 05:42:51 2017

@author: Nitin
"""

class Human:
    def __init__(self,name,age,sex):
        self.changeName(name)
        self.changeAge(age)
        self.sex = sex
    def changeName(self,name):
        self.name = name + ' Mahajan'
    def changeAge(self,age):
        self.age  = age + 10
    def get(self,name,age,res):
        
        print("My name is:")
        print(name)
        print("My age  is :")
        print(age)
#        print("My sex  is:")
#        print(sex)
        print("-----------------------")
        print("My name is(Human.name):")
        print(self.name)
        print("My age  is (Human.age):")
        print(self.age)
        print("My sex  is(Human.sex):")
        print(self.sex)

nm = Human('Anu' , 200 , 'not disclosed')
nm.get('Annu' , 10 ,'Pickering')
            
            