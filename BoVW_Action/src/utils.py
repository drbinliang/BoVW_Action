'''
Created on 29/05/2014

@author:  Bin Liang
@email: bin.liang.ty@gmail.com
'''
import config

def getCategoryIdByName(categoryName):
    """ Get category id by category name """
    return config.actionCateogory.index(categoryName)