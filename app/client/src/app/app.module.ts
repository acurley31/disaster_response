import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';

import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { SharedModule } from "./shared.module";
import { MessageInputComponent } from './components/message-input/message-input.component';
import { ClassifierComponent } from './components/classifier/classifier.component';
import { MessagePipe } from './pipes/message.pipe';


@NgModule({
  declarations: [
    AppComponent,
    MessageInputComponent,
    ClassifierComponent,
    MessagePipe
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    BrowserAnimationsModule,
    SharedModule,
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
