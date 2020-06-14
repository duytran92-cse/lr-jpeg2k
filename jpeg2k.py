

# import skikit-image
import JP2000.Functions as jp2
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage as ndsig
import numpy.linalg as nl


### change all img ---> techno

techno = np.double(plt.imread('./technoforum.jpg'))		
tailleX = techno.shape[0]
tailleY = techno.shape[1]

# Affecte a la variable LO_D le filtre H de decomposition
LO_D=[1/np.sqrt(2),1/np.sqrt(2)]

# Affecte a la variable HI_D le filtre G de decomposition
HI_D=[-1/np.sqrt(2),1/np.sqrt(2)]
print(LO_D, HI_D)

def decomposer(techno, LO_D, HI_D, compteur):
    compteur -= 1
    if(compteur == 0):
        a1 = jp2.decompose(LO_D, LO_D, techno)
    else:
        a1 = decomposer(jp2.decompose(LO_D, LO_D, techno), LO_D, HI_D, compteur)
    im_out=np.double(np.zeros(techno.shape))
    d1 = jp2.decompose(HI_D, HI_D, techno)
    h1 = jp2.decompose(LO_D, HI_D, techno)
    v1= jp2.decompose(HI_D, LO_D, techno)
    print("techno shape : ",techno.shape)
    print("h1 shape : ",h1.shape)
    
    im_out[0:a1.shape[0],0:a1.shape[1]] = a1
    im_out[0:a1.shape[0],a1.shape[0]:2*a1.shape[0]] = h1
    im_out[a1.shape[0]:2*a1.shape[0],0:a1.shape[1]] = v1
    im_out[a1.shape[0]:2*a1.shape[0],a1.shape[0]:2*a1.shape[0]] = d1

    # compress image
    #plt.imshow(jp2.composite_image(a1,np.abs(v1),np.abs(h1),np.abs(d1)),cmap='gray')
    #plt.show()

    h2=jp2.decompose(HI_D,LO_D,a1)
    v2=jp2.decompose(LO_D,HI_D,a1)
    d2=jp2.decompose(HI_D,HI_D,a1)
    a2=jp2.decompose(LO_D,LO_D,a1)

    # plt.imshow(jp2.composite_image(a2*0.1,np.abs(v2),np.abs(h2),np.abs(d2)),cmap='gray')
    # plt.show()

    return im_out

# """

# decompress image
# plt.imshow(decomposer(techno, LO_D, HI_D, 2), cmap='gray')
# plt.show()
# """


techno_compressee = decomposer(techno, LO_D, HI_D, 2)
#Charge dans la variable LO_R le filtre H tilde de reconstruction
LO_R=[1/np.sqrt(2),1/np.sqrt(2)]
# Charge dans la variable HI_R le filtre G tilde de reconstruction
HI_R=[1/np.sqrt(2),-1/np.sqrt(2)]

def recomposer(techno, LO_R, HI_R, compteur):
    compteur -= 1
    tailleX = int(techno.shape[0]/2)
    tailleY = int(techno.shape[1]/2)
    a1 = techno[0:tailleX,0:tailleY]
    h1 = techno[0:tailleX,tailleY:2*tailleY]
    v1 = techno[tailleX:2*tailleX,0:tailleY]
    d1 = techno[tailleX:2*tailleX,tailleY:2*tailleY]
    if(compteur == 0):
        a1_r = jp2.reconstruction(LO_R, LO_R,a1)
    else:
        a1_r = jp2.reconstruction(LO_R, LO_R,recomposer(a1, LO_R, HI_R, compteur))
    h1_r = jp2.reconstruction(LO_R, HI_R,h1)
    v1_r = jp2.reconstruction(HI_R, LO_R,v1)
    d1_r = jp2.reconstruction(HI_R, HI_R,d1)
    techno = a1_r + h1_r + v1_r + d1_r
    return techno

techno_recompose = recomposer(techno_compressee, LO_R, HI_R, 2)
print("erreur de reconstruction : ",np.sum(abs(techno_recompose-techno)))

plt.imshow(techno_recompose, cmap='gray')
plt.show()

err = nl.norm(techno_recompose-np.double(techno),2)
print('Erreur de reconstruction SANS SEUILLAGE sur les coefficients:',err)




lena = np.double(plt.imread('./Lena.jpg'))


lena_compressee = decomposer(lena, LO_D, HI_D, 2)
lena_recomposee = recomposer(lena_compressee, LO_R, HI_R, 2)

plt.subplot(1,3,1)
plt.imshow(lena, cmap='gray')
plt.subplot(1,3,2)
plt.imshow(lena_compressee, cmap='gray')
plt.subplot(1,3,3)
plt.imshow(lena_compressee, cmap='gray')
plt.show()


thres=100
v2 = [120, 100]
v2_s = np.where(np.abs(v2) > thres, v2, 0.0)
print(v2_s)

def decomposerSeuil(img, LO_D, HI_D, compteur, seuil):
    compteur -= 1
    #print(img)
    if(compteur == 0):
        a1 = jp2.decompose(LO_D, LO_D, img)
    else:
        a1 = decomposerSeuil(jp2.decompose(LO_D, LO_D, img), LO_D, HI_D, compteur, seuil)
    im_out=np.double(np.zeros(img.shape))
    d1 = jp2.decompose(HI_D, HI_D, img)
    h1 = jp2.decompose(LO_D, HI_D, img)
    v1= jp2.decompose(HI_D, LO_D, img)
    #pour seuiller les ondelettes: :
    h1 = np.where(np.abs(h1) > seuil, h1, 0.0)
    v1 = np.where(np.abs(v1) > seuil, v1, 0.0)
    d1 = np.where(np.abs(d1) > seuil, d1, 0.0)
    
    im_out[0:a1.shape[0],0:a1.shape[1]] = a1
    im_out[0:a1.shape[0],a1.shape[0]:2*a1.shape[0]] = h1
    im_out[a1.shape[0]:2*a1.shape[0],0:a1.shape[1]] = v1
    im_out[a1.shape[0]:2*a1.shape[0],a1.shape[0]:2*a1.shape[0]] = d1
    return im_out

def decomposerSeuilTaille(img, LO_D, HI_D, compteur, seuil, taille, energie):
    compteur -= 1
    #print(img)
    if(compteur == 0):
        a1 = jp2.decompose(LO_D, LO_D, img)
    else:
        a1, count, energie = decomposerSeuilTaille(jp2.decompose(LO_D, LO_D, img), LO_D, HI_D, compteur, seuil, taille, energie)
        taille += count 
        energie += energie
    im_out=np.double(np.zeros(img.shape))
    d1 = jp2.decompose(HI_D, HI_D, img)
    h1 = jp2.decompose(LO_D, HI_D, img)
    v1= jp2.decompose(HI_D, LO_D, img)
    #pour seuiller les ondelettes: :
    h1 = np.where(np.abs(h1) > seuil, h1, 0.0)
    v1 = np.where(np.abs(v1) > seuil, v1, 0.0)
    d1 = np.where(np.abs(d1) > seuil, d1, 0.0)
    taille += (h1.size - np.count_nonzero(h1) +
    v1.size - np.count_nonzero(v1) +
    d1.size - np.count_nonzero(d1) )
    energie += ( (h1**2).sum() + (v1**2).sum() + (d1**2).sum() )
    
    im_out[0:a1.shape[0],0:a1.shape[1]] = a1
    im_out[0:a1.shape[0],a1.shape[0]:2*a1.shape[0]] = h1
    im_out[a1.shape[0]:2*a1.shape[0],0:a1.shape[1]] = v1
    im_out[a1.shape[0]:2*a1.shape[0],a1.shape[0]:2*a1.shape[0]] = d1
    return im_out, taille, energie

def calulEnergie(img):
    cpt = 2
    imgCompressee, taille1, energie1 = decomposerSeuilTaille(img, LO_D, HI_D, cpt, 20, 0, 0)
    imgRecomposee = recomposer(imgCompressee, LO_R, HI_R, cpt)
    # plt.subplot(1,4,1)
    # plt.imshow(imgRecomposee,cmap='gray')

    imgCompressee, taille2, energie2 = decomposerSeuilTaille(img, LO_D, HI_D, cpt, 50, 0, 0)
    imgRecomposee = recomposer(imgCompressee, LO_R, HI_R, cpt)
    # plt.subplot(1,4,2)    
    # plt.imshow(imgRecomposee,cmap='gray')

    imgCompressee, taille3, energie3 = decomposerSeuilTaille(img, LO_D, HI_D, cpt, 100, 0, 0)
    imgRecomposee = recomposer(imgCompressee, LO_R, HI_R, cpt)
    # plt.subplot(1,4,3)
    # plt.imshow(imgRecomposee,cmap='gray')

    imgCompressee, taille4, energie4 = decomposerSeuilTaille(img, LO_D, HI_D, cpt, 200, 0, 0)
    imgRecomposee = recomposer(imgCompressee, LO_R, HI_R, cpt)
    # plt.subplot(1,4,4)
    # plt.imshow(imgRecomposee,cmap='gray')
    # plt.show()

    listeTailles = []
    listeEnergies = []
    listeTailles += (taille1, taille2, taille3, taille4)
    listeEnergies += (energie1, energie2, energie3, energie4)
    listeTailles = np.array(listeTailles)
    listeEnergies = np.array(listeEnergies)
    return listeTailles, listeEnergies

img = np.double(plt.imread('./technoforum.jpg'))

plt.subplot(2,2,1)
plt.imshow(img, cmap='gray')
listeTailles, listeEnergies = calulEnergie(img)
plt.subplot(2,2,3)
plt.plot(listeTailles, listeEnergies)


### lena
img = np.double(plt.imread('./Lena.jpg'))
plt.subplot(2,2,2)
plt.imshow(img, cmap='gray')
listeTailles, listeEnergies = calulEnergie(img)
plt.subplot(2,2,4)
plt.plot(listeTailles, listeEnergies)
plt.show()