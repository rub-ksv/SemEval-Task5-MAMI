import os, sys
import cv2
import numpy as np

def main(mainsrcdir, dset):
    srcdir = os.path.join(mainsrcdir, dset)
    savedir = os.path.join(mainsrcdir, dset + '_split')
    threshold = 0.7
    varthreshold = 100
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    filelist = os.listdir(srcdir)
    filelist.remove(dset + '.csv')
    for i in filelist:
        print(i)
        savefilename = os.path.join(savedir, i.strip('.jpg'))
        if not os.path.exists(savefilename):
            os.makedirs(savefilename)
        infiledir = os.path.join(srcdir, i)
        img = cv2.imread(infiledir)
        grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        imageshape = np.shape(grey)
        yaxis = []
        for i in range(imageshape[0] - 1):
            corr = np.corrcoef(grey[i, :], grey[i + 1, :])
            yaxis.append(corr[0, 1])
        ycutpoint = []
        for i in range(len(yaxis)):
            if yaxis[i] < threshold:
                ycutpoint.append(i)
        ycutpoint.insert(0, 0)
        ycutpoint.append(imageshape[0])
        splitdatay = []
        for i in range(len(ycutpoint) - 1):
            splitdatay.append(img[ycutpoint[i]: ycutpoint[i + 1], :, :])
        splitdataysave = []
        for i in range(len(splitdatay)):
            datashape = np.shape(splitdatay[i])
            if datashape[0] < 100:
                pass
            elif datashape[1] < 100:
                pass
            else:
                splitdataysave.append(splitdatay[i])
        outlist = []
        for j in splitdataysave:
            grey = (0.299 * j[:, :, 2] + 0.587 * j[:, :, 1] + 0.114 * j[:, :, 0]).astype(int)
            xaxis = []
            for i in range(imageshape[1] - 1):
                corr = np.corrcoef(grey[:, i], grey[:, i + 1])
                xaxis.append(corr[0, 1])
            xcutpoint = []
            for i in range(len(xaxis)):
                if xaxis[i] < threshold:
                    xcutpoint.append(i)
            xcutpoint.insert(0, 0)
            xcutpoint.append(imageshape[1])
            splitdatax = []
            for i in range(len(xcutpoint) - 1):
                splitdatax.append(j[:, xcutpoint[i]: xcutpoint[i + 1], :])
            splitdata = []
            for i in range(len(splitdatax)):
                datashape = np.shape(splitdatax[i])
                if datashape[0] < 100:
                    pass
                elif datashape[1] < 100:
                    pass
                else:
                    splitdata.append(splitdatax[i])
            outlist.append(splitdata)
            for i in range(len(outlist)):
                for j in range(len(outlist[i])):
                    savefiledir = os.path.join(savefilename, str(i) + '_' + str(j) + '.jpg')
                    try:
                        greyout = (0.299 * outlist[i][j][:, :, 2] + 0.587 * outlist[i][j][:, :, 1] + 0.114 * outlist[i][j][:, :, 0]).astype(int)
                        imagevar = greyout.var()
                        if imagevar < varthreshold:
                            pass
                        else:
                            cv2.imwrite(savefiledir, outlist[i][j])
                    except:
                        pass

            cv2.imwrite(os.path.join(savefilename, 'org.jpg'), img)


# hand over parameter overview
# sys.argv[1] = mainsrcdir (str): Dataset dir
# sys.argv[2] = dset (str): Which set.
main(sys.argv[1],sys.argv[2])

