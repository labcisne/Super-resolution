#Importando as bibliotecas necessárias
import cv2 as cv
import imutils
import numpy as np
import random as rd
import os
import csv
import secrets

#Criação da função
def GenerateLowResolution(folder, qty = 5, rf=2, k=(3,3), theta=(0,1), tx=(-10,10), ty=(-10,10)):
    
    #Varredura das imagens na pasta
    for file in os.listdir(folder):

        #Leitura da imagem
        img = cv.imread(os.path.join(folder,file))   
        
        #Validação das imagens lidas
        if img is not None:

            #Criação da pasta LR
            folder_name = f'LR - {file}'
            path=os.path.join(folder,folder_name)

            #Verificação da existência da pasta
            if os.path.exists(path) == False:
                #Criação da Pasta LR quando ainda não existir
                os.makedirs(path)     

            for i in range(0,qty):

                #Leitura das dimensões da imagem
                height = img.shape[0]
                width = img.shape[1]

                #Definição dos novos tamanhos da imagem (divisão inteira)
                nheight = height//rf
                nwidth = width//rf

                #Redimensionamento 1: Alterando as dimensões da imagem usando os novos tamanhos
                nimg = cv.resize(img, (nwidth,nheight))

                #Aplicação de filtros de borramento
                nimgG= cv.GaussianBlur(nimg, k, 0)

                #Gerando aleatóriamente o ângulo (em graus)
                ang = rd.uniform(theta[0],theta[1])

                #Rotacionando as imagens e 
                rotated = imutils.rotate(nimgG, angle=ang)
                
                #Gerando aleatóriamente as componentes de tranlação em y e em x
                transx = rd.randint(tx[0], ty[1])
                transy = rd.randint(ty[0], ty[1])

                #Definindo a matriz de translação 
                M = np.float32([[1, 0, transx], [0, 1, transy]])

                #Transladando a imagem
                shifted = cv.warpAffine(rotated, M, (rotated.shape[1], rotated.shape[0]))

                #Criação do id hexadecimal aleatório
                id = secrets.token_hex(3)
            
                #Salvando a imagem em uma pasta existente
                cv.imwrite(f"{path}/{id}.png",shifted)
                
                #Diretório do arquivo csv
                arq_csv = 'LR.csv'
                            
                #Verificação da existência do arquivo
                if os.path.exists(arq_csv) == True:
                
                    #Adição de dados no arquivo
                    csv_file = open('LR.csv', 'a',newline = '')
                    writer = csv.writer(csv_file)
                    writer.writerow([id, file, rf, k, ang, transx, transy])

                else:

                    #Criação do arquivo e adição dos dados
                    with open('LR.csv', 'w', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        field = ["id", "original image", "rf", "k", "theta", "tx", "ty"]
                        writer.writerow(field)
                        writer.writerow([id, file, rf, k, ang, transx, transy])

if __name__ == "__main__":
    GenerateLowResolution("imagens")
