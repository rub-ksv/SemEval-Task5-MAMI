import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from matplotlib.ticker import MaxNLocator

def plotimage2model(srcdir, type):
        plt.clf()
        nice_fonts = {
                # Use LaTeX to write all text
                # Use 10pt font in plots, to match 10pt font in document
                "axes.labelsize": 15,
                "font.size": 15,
                # Make the legend/label fonts a little smaller
                "legend.fontsize": 20,
                "xtick.labelsize": 15,
                "ytick.labelsize": 15,
        }

        matplotlib.rcParams.update(nice_fonts)

        accdict = {}

        accdict.update({type: {}})
        acclist = os.listdir(os.path.join(srcdir, type))
        accs = []
        for i in acclist:
                accdict[type].update({i: {}})
                filenames = os.listdir(os.path.join(srcdir, type, i))
                for filename in filenames:
                        if filename.endswith('.pkl'):
                                fold = filename.split('_')[0]
                                acc = float(filename.split('_')[-1].strip('.pkl'))
                                accdict[type][i].update({fold: acc})

        plt.ylabel('acc (%)')
        plt.xlabel('weight')

        x = np.arange(0, 1, 0.1)
        acc = {}
        for xid in range(10):
                accout = []
                for key in x:
                        key = round(key, 2)
                        accout.append(accdict[type][str(key)][str(xid)])
                acc.update({xid: accout})
        for key in list(acc.keys()):
                plt.plot(x, acc[key], label=key)

        # plt.ylim(0.0, 55.0)

        plt.legend(loc='lower right', ncol=1, prop={'size': 5})
        plt.tick_params(axis='x', pad=8)
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.savefig(os.path.join(srcdir, type + '_ensemble.pdf'), bbox_inches="tight")








def plotimage2model_task2(srcdir):
        typelist = ['mean', 'misogynous', 'shaming', 'stereotype', 'objectification', 'violence', 'all']
        for typeid in range(len(typelist)):
                plt.figure()
                nice_fonts = {
                        # Use LaTeX to write all text
                        # Use 10pt font in plots, to match 10pt font in document
                        "axes.labelsize": 15,
                        "font.size": 15,
                        # Make the legend/label fonts a little smaller
                        "legend.fontsize": 20,
                        "xtick.labelsize": 15,
                        "ytick.labelsize": 15,
                }

                matplotlib.rcParams.update(nice_fonts)

                x = np.arange(0, 1, 0.1)
                typestemp = os.listdir(srcdir)
                types = []
                for c in range(len(typestemp)):
                        if '.pdf' in typestemp[c]:
                                pass
                        else:
                                types.append(typestemp[c])


                accdict = {}
                for type in types:
                        acclist = os.listdir(os.path.join(srcdir, type))
                        accs = []
                        for i in acclist:
                                if i.endswith('.pkl'):
                                        accs.append(float(i.split('_')[typeid + 2].strip('.pkl')))
                        accdict.update({type: accs})

                plt.ylabel('acc (%)')
                plt.xlabel('weight')

                for key in list(accdict.keys()):
                        plt.plot(x, accdict[key], label=key)

                # plt.ylim(0.0, 55.0)

                plt.legend(loc='lower right', ncol=1, prop={'size': 5})
                plt.tick_params(axis='x', pad=8)
                plt.grid(b=True, which='major', color='#666666', linestyle='-')
                plt.savefig(os.path.join(srcdir, typelist[typeid] + '.pdf'), bbox_inches="tight")


def compare2models(srcdir, stream, type1, type2):
        '''nice_fonts = {
                # Use LaTeX to write all text
                # Use 10pt font in plots, to match 10pt font in document
                "axes.labelsize": 15,
                "font.size": 15,
                # Make the legend/label fonts a little smaller
                "legend.fontsize": 20,
                "xtick.labelsize": 15,
                "ytick.labelsize": 15,
        }

        matplotlib.rcParams.update(nice_fonts)'''
        plt.figure()
        x = np.arange(0, 10, 1)
        models = os.listdir(srcdir)
        accdict = {}
        for j in models:
                try:
                        if j.startswith(stream):
                                savename = '_'.join(j.split(' ')[:2])
                                subfiles = os.listdir(os.path.join(srcdir, j))
                                model = '_'.join([type1, type2])
                                acclist = os.listdir(os.path.join(srcdir, j, subfiles[0], model))
                                accs = []
                                for i in acclist:
                                        if i.endswith('.pkl'):
                                                accs.append(float(i.split('_')[-1].strip('.pkl')))
                                accdict.update({savename: accs})
                except:
                        pass





        plt.ylabel('acc (%)')
        plt.xlabel('weight')

        for key in list(accdict.keys()):
                plt.plot(x, accdict[key], label=key)

        #plt.ylim(0.0, 55.0)

        plt.legend(loc='lower right', ncol=1, prop={'size': 5})
        plt.tick_params(axis='x', pad=8)
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.savefig(os.path.join(srcdir, 'ensemble.pdf'), bbox_inches="tight")

def compare2models_task2(srcdir, stream, type1, type2):
        '''nice_fonts = {
                # Use LaTeX to write all text
                # Use 10pt font in plots, to match 10pt font in document
                "axes.labelsize": 15,
                "font.size": 15,
                # Make the legend/label fonts a little smaller
                "legend.fontsize": 20,
                "xtick.labelsize": 15,
                "ytick.labelsize": 15,
        }

        matplotlib.rcParams.update(nice_fonts)'''
        typelist = ['mean', 'misogynous', 'shaming', 'stereotype', 'objectification', 'violence', 'all']
        for typeid in range(len(typelist)):
                plt.figure()
                x = np.arange(0, 10, 1)
                models = os.listdir(srcdir)
                accdict = {}
                for j in models:
                        try:
                                if j.startswith(stream):
                                        savename = '_'.join(j.split(' ')[:2])
                                        subfiles = os.listdir(os.path.join(srcdir, j))
                                        model = '_'.join([type1, type2])
                                        acclist = os.listdir(os.path.join(srcdir, j, subfiles[0], model))
                                        accs = []
                                        for i in acclist:
                                                if i.endswith('.pkl'):
                                                        accs.append(float(i.split('_')[-1].strip('.pkl')))
                                        accdict.update({savename: accs})
                        except:
                                pass

                plt.ylabel('acc (%)')
                plt.xlabel('weight')

                for key in list(accdict.keys()):
                        plt.plot(x, accdict[key], label=key)

                # plt.ylim(0.0, 55.0)

                plt.legend(loc='lower right', ncol=1, prop={'size': 5})
                plt.tick_params(axis='x', pad=8)
                plt.grid(b=True, which='major', color='#666666', linestyle='-')
                plt.savefig(os.path.join(srcdir, typelist[typeid] + '.pdf'), bbox_inches="tight")


def plotimage3model(srcdir):
        '''nice_fonts = {
                # Use LaTeX to write all text
                # Use 10pt font in plots, to match 10pt font in document
                "axes.labelsize": 15,
                "font.size": 15,
                # Make the legend/label fonts a little smaller
                "legend.fontsize": 20,
                "xtick.labelsize": 15,
                "ytick.labelsize": 15,
        }

        matplotlib.rcParams.update(nice_fonts)'''
        plt.figure()
        x = np.arange(0, 55, 1)
        types = os.listdir(srcdir)
        accdict = {}
        for type in types:
                acclist = os.listdir(os.path.join(srcdir, type))
                accs = []
                for i in acclist:
                        if i.endswith('.pkl'):
                                accs.append(float(i.split('_')[-1].strip('.pkl')))
                accdict.update({type: accs})

        plt.ylabel('acc (%)')
        plt.xlabel('weight index')

        for key in list(accdict.keys()):
                plt.plot(x, accdict[key], label=key)

        #plt.ylim(0.0, 55.0)

        plt.legend(loc='lower right', ncol=1, prop={'size': 5})
        plt.tick_params(axis='x', pad=8)
        plt.grid(b=True, which='major', color='#666666', linestyle='-')
        plt.savefig(os.path.join(srcdir, 'ensemble.pdf'), bbox_inches="tight")

def plotimage3model_task2(srcdir):
        '''nice_fonts = {
                # Use LaTeX to write all text
                # Use 10pt font in plots, to match 10pt font in document
                "axes.labelsize": 15,
                "font.size": 15,
                # Make the legend/label fonts a little smaller
                "legend.fontsize": 20,
                "xtick.labelsize": 15,
                "ytick.labelsize": 15,
        }

        matplotlib.rcParams.update(nice_fonts)'''
        typelist = ['mean', 'misogynous', 'shaming', 'stereotype', 'objectification', 'violence', 'all']
        for typeid in range(len(typelist)):
                plt.figure()
                x = np.arange(0, 55, 1)
                typestemp = os.listdir(srcdir)
                types = []
                for c in range(len(typestemp)):
                        if '.pdf' in typestemp[c]:
                                pass
                        else:
                                types.append(typestemp[c])
                accdict = {}
                for type in types:
                        acclist = os.listdir(os.path.join(srcdir, type))
                        accs = []
                        for i in acclist:
                                if i.endswith('.pkl'):
                                        accs.append(float(i.split('_')[typeid + 4].strip('.pkl')))
                        accdict.update({type: accs})

                plt.ylabel('acc (%)')
                plt.xlabel('weight index')

                for key in list(accdict.keys()):
                        plt.plot(x, accdict[key], label=key)

                # plt.ylim(0.0, 55.0)

                plt.legend(loc='lower right', ncol=1, prop={'size': 5})
                plt.tick_params(axis='x', pad=8)
                plt.grid(b=True, which='major', color='#666666', linestyle='-')
                plt.savefig(os.path.join(srcdir, typelist[typeid] + '.pdf'), bbox_inches="tight")


def plotimagemultimodel(srcdir):
        y = []
        x = []
        acclist = os.listdir(srcdir)
        for i in acclist:
                if i.endswith('.pkl'):
                        splits = i.split('_all_')
                        x.append(splits[0])
                        y.append(float(splits[1].strip('.pkl')))

        plt.barh(x, y)

        for index, value in enumerate(y):
                plt.text(value, index, str(value))
        plt.savefig(os.path.join(srcdir, 'ensemble.pdf'), bbox_inches="tight")
