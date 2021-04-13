#ifndef SOFA_GUI_QT_SURFLABLOGIN_H
#define SOFA_GUI_QT_SURFLABLOGIN_H

#include <QDialog>
#include <QString>
#include <windows.h>
#include "SofaGuiQt.h"
#include <ui_surflablogin.h>

class Ui_SurfLabLogin;

namespace sofa
{
	namespace gui
	{
		namespace qt
		{

			class SOFA_SOFAGUIQT_API SurfLabLogin : public QDialog
			{
				Q_OBJECT
					//std::unique_ptr<Ui_SurfLabLogin> ui;
					Ui_SurfLabLogin* ui;

			public:
				SurfLabLogin(QWidget* parent = 0);
				~SurfLabLogin();
				QString studentEmail;
				QString studentName;
				QString destinationEmail;
				static SurfLabLogin* getInstance();
			private slots:

				void on_ButtonBox_clicked(QAbstractButton* button);
			};
			//SurfLabLogin* SurfLabLogin::instance = 0;
		}
	}
}
#endif // SOFA_GUI_QT_SURFLABLOGIN_H