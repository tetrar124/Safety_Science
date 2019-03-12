from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import os
import sys
import glob
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
import pylab as plt
import math
from networkx import *
import numpy as np
import cairosvg

class NewFingerprint(object):

def extSpecialTox(self):
    #急性毒性
    # os.chdir(r'G:\マイドライブ\Data\tox_predict\all_data')
    # df = pd.read_csv('structure_result.csv', engine='python', encoding='cp932')
    #慢性毒性
    os.chdir(r'G:\マイドライブ\Data\Meram Chronic Data')
    df = pd.read_csv('extChronicStrcture.csv')

    #df = df[['CAS', '化学物質名', '毒性値', '暴露時間（日）', '生物種', '栄養段階']]
    #カーバメート
    #carbamateDf = df[df['CAS'].isin(['22781-23-3','16752-77-5','112410-23-8',\
                             # '161050-58-4','79127-80-3','82560-54-1',\
                             # '65907-30-4','114-26-1','55285-14-8','63-25-2',\
                             # '1129-41-5','116-06-3','143807-66-3','23103-98-2',\
                             # '2631-40-5','3766-81-2','1563-66-2','59669-26-0',\
                             # '29973-13-5','28217-97-2','2686-99-9','78-57-9'])]
    #carbameteSMILES = carbamateDf['CAS','canonical_smiles']
    # 有機リン
    # exDfP = df[df['CAS'].isin(['115-86-6', '126-73-8', '141-66-2', '298-07-7', '300-76-5',
    #         '34364-42-6', '62-73-7', '107-66-4', '1241-94-7', '13171-21-6',
    #         '2528-36-1', '29761-21-5', '311-45-5', '3689-24-5', '470-90-6',
    #         '598-02-7', '68333-79-9', '6923-22-4', '7722-88-5', '96300-97-9',
    #         '10042-91-8', '107-49-3', '512-56-1', '5598-15-2', '7558-80-7',
    #         '7601-54-9', '7783-28-0', '96300-95-7'])]

    #carbameteSMILES.to_csv('carbameteSMILES.csv')
    #カーバメート
    carbamateInsets=['OC(=O)NC([H])([H])[H]','C([H])([H])([H])NC(=O)O',
                     'NN(C(=O))C(C([H])([H])[H])(C([H])([H])[H])C([H])([H])[H]','NC(=S)N',
                     'C([H])([H])([H])COC(=O)NC','NSN']
    #有機リン
    organicPhosphoricAcids =['CO[P](=S)(OC)OC','CO[P](=O)(NC)OC', 'CO[P](=O)(OC)OC',
                            'CO[P](=O)(SC)OC','CO[P](=O)(SC)N','CO[P](=S)(SC)N',
                            'CO[P](=S)(SC)OC']
    #Metal
    metals = ['Li','Be','Na','Mg','Al','K','Ca','Sc','Ti','V','Cr','Mn','Fe','Co','Ni',
                'Cu','Zn','Ga','Ge','As','Se','Rb','Sr','Y','Zr','Nb','Mo',
                'Tc','Ru','Rh','Pd','Ag','Cd','In','Sn','Sb','Te','Cs','Ba',
                'Hf','Ta','W','Re','Os','Ir','Pt','Au','Hg','Tl','Pb','Bi','Po','Cr+6']
    #有機塩素
    organicCls = ['CCl', 'ClC','Cl)','NCl','ClN','SCl','ClS',')Cl']
    #有機ヒ素
    organicAss = ['C[As','N[As','S[As']
    #有機スズ
    organicSns = ['C[Sn','N[Sn','S[Sn']
    columns = ['CAS'] + np.arange(0,60,1).tolist()
    resultDf = pd.DataFrame(columns=columns)
    for i,(cas,smiles) in enumerate(zip(df['CAS'],df['canonical_smiles'])):
        if pd.isnull(smiles)==True:
            pass

        else:
            base_mol = Chem.MolFromSmiles(smiles)
            if base_mol == None:
                pass
            else:
                keys = [cas]
                #メタルチェック
                for metalName in metals:
                    if smiles.count(metalName) > 0:
                        keys.append(1)
                        print('金属',smiles,metalName)
                    else:
                        keys.append(0)
                #電離
                if smiles.count('.') > 0:
                    keys.append(1)
                    print('金属', smiles, metalName)
                else:
                    keys.append(0)

                #有機塩素
                templist = []
                for organicCl in organicCls:
                    if smiles.count(organicCl) > 0:
                        templist.append(1)
                if sum(templist) > 0:
                    print('有機塩素', smiles)
                    keys.append(1)
                else:
                    keys.append(0)
                #有機ヒ素
                templist = []
                for organicAs in organicAss:
                    if smiles.count(organicAs) > 0:
                        templist.append(1)
                if sum(templist) > 0:
                    print('有機ヒ素', smiles)
                    keys.append(1)
                else:
                    keys.append(0)
                #有機スズ
                templist = []
                for organicSn in organicSns:
                    if smiles.count(organicSn) > 0:
                        templist.append(1)
                if sum(templist) > 0:
                    print('有機スズ', smiles)
                    keys.append(1)
                else:
                    keys.append(0)
                #カーバメート
                templist = []
                for carbamate in carbamateInsets:
                    tmpCar = Chem.MolFromSmiles(carbamate)
                    ph_mols = base_mol.HasSubstructMatch(tmpCar)
                    if ph_mols == True:
                        print('カーバメート殺虫剤',i,smiles,ph_mols)
                        templist.append(1)
                if sum(templist) > 0:
                    keys.append(1)
                else:
                    keys.append(0)
                #有機リン
                templist = []
                for organicP in organicPhosphoricAcids:
                    tmpCar = Chem.MolFromSmiles(organicP)
                    ph_mols = base_mol.HasSubstructMatch(tmpCar)
                    if ph_mols == True:
                        print('有機リン',i,smiles,ph_mols)
                        templist.append(1)
                if sum(templist) > 0:
                    keys.append(1)
                else:
                    keys.append(0)
                #ジフェニルエーテル
                tmpCar = Chem.MolFromSmiles('C1=C(OC2=CC=CC=C2)C=CC=C1')
                ph_mols = base_mol.HasSubstructMatch(tmpCar)
                if ph_mols == True:
                    keys.append(1)
                else:
                    keys.append(0)

                tempDf = pd.DataFrame(data = [keys],columns=columns)
                resultDf = pd.concat([resultDf,tempDf])
    resultDf.to_csv('newFingerprint.csv',index=False)

    for i,  smiles in enumerate(carbamateInsets):
    #for  i ,smiles in enumerate(organicPhosphoricAcids):
        m = Chem.MolFromSmiles(smiles)
        view = rdMolDraw2D.MolDraw2DSVG(200, 200)
        tm = rdMolDraw2D.PrepareMolForDrawing(m)
        option = view.drawOptions()
        option.circleAtoms = False
        option.continuousHighlight = False
        view.DrawMolecule(tm)
        view.FinishDrawing()
        svg = view.GetDrawingText()
        name = './' + str(i) + 'cabamate' + '.png'
        with open('out.svg', 'w') as f:
            f.write(svg)
        cairosvg.svg2png(url='out.svg', write_to=name)
        #SVG(svg.replace('svg:', ''))


if __name__ == '__main__':
    new = NewFingerprint()
    new.extSpecialTox()